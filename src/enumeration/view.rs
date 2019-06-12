//! View Enumeration test
//!
//! Tests the row assignment algorithms separately from the column algorithms.
use std::collections::BTreeMap;

use braid_utils::misc::logsumexp;
use rand::Rng;

use crate::cc::assignment::lcrp;
use crate::cc::transition::ViewTransition;
use crate::cc::{
    AssignmentBuilder, ColModel, Feature, RowAssignAlg, View, ViewBuilder,
};
use crate::enumeration::{
    build_features, normalize_assignment, partition_to_ix,
};
use crate::misc::Partition;

/// Compute the posterior of all assignments of the features under CRP(alpha)
///
/// NOTE: The rng is required, for calling AssignmentBuilder.build, but nothing
/// random should actually happen.
fn calc_partition_ln_posterior<R: Rng>(
    features: &Vec<ColModel>,
    alpha: f64,
    mut rng: &mut R,
) -> BTreeMap<u64, f64> {
    let n = features[0].len();
    let mut ln_posterior: BTreeMap<u64, f64> = BTreeMap::new();

    Partition::new(n).for_each(|z| {
        let ix = partition_to_ix(&z);

        // NOTE: We don't need seed control here because both alpha and the
        // assignment are set, but I'm setting the seed anyway in case the
        // assignment builder internals change
        let asgn = AssignmentBuilder::from_vec(z)
            .with_alpha(alpha)
            .from_rng(&mut rng)
            .build()
            .unwrap();

        let ln_pz = lcrp(n, &asgn.counts, alpha);

        let view: View = ViewBuilder::from_assignment(asgn)
            .with_features(features.clone())
            .from_rng(&mut rng)
            .build();

        ln_posterior.insert(ix, view.score() + ln_pz);
    });

    ln_posterior
}

/// Normalize and exp the posterior computed in calc_partition_ln_posterior
fn norm_posterior(ln_posterior: &BTreeMap<u64, f64>) -> BTreeMap<u64, f64> {
    let logps: Vec<f64> = ln_posterior.values().map(|&p| p).collect();
    let z = logsumexp(&logps);
    let mut normed: BTreeMap<u64, f64> = BTreeMap::new();
    for (key, lp) in ln_posterior {
        normed.insert(*key, (lp - z).exp());
    }
    normed
}

/// Compute the sum absolute error (in 0 to 1), for a View consisting of `ncols`
/// continuous columns each with `nrows` rows.
pub fn view_enum_test(
    nrows: usize,
    ncols: usize,
    n_runs: usize,
    n_iters: usize,
    row_alg: RowAssignAlg,
) -> f64 {
    let mut rng = rand::thread_rng();
    let features = build_features(nrows, ncols, &mut rng);
    let ln_posterior = calc_partition_ln_posterior(&features, 1.0, &mut rng);
    let posterior = norm_posterior(&ln_posterior);

    let transitions: Vec<ViewTransition> = vec![
        ViewTransition::RowAssignment,
        ViewTransition::ComponentParams,
    ];

    let mut est_posterior: BTreeMap<u64, f64> = BTreeMap::new();
    let inc: f64 = ((n_runs * n_iters) as f64).recip();

    for _ in 0..n_runs {
        let asgn = AssignmentBuilder::new(nrows)
            .with_alpha(1.0)
            .from_rng(&mut rng)
            .build()
            .unwrap();
        let mut view = ViewBuilder::from_assignment(asgn)
            .with_features(features.clone())
            .from_rng(&mut rng)
            .build();
        for _ in 0..n_iters {
            view.update(10, row_alg, &transitions, &mut rng);

            let normed = normalize_assignment(view.asgn.asgn.clone());
            let ix = partition_to_ix(&normed);

            if !posterior.contains_key(&ix) {
                println!("{:?}", view.asgn.asgn);
                println!("{:?}", normed);
                panic!("invalid index!");
            }

            *est_posterior.entry(ix).or_insert(0.0) += inc;
        }
    }

    assert!(!est_posterior.keys().any(|k| !posterior.contains_key(k)));

    let mut cdf = 0.0;
    let mut est_cdf = 0.0;
    posterior.iter().fold(0.0, |err, (key, &p)| {
        cdf += p;
        if est_posterior.contains_key(key) {
            est_cdf += est_posterior[key];
        }
        err + (cdf - est_cdf).abs()
    }) / posterior.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_TRIES: u32 = 5;

    fn flaky_test_passes<F>(n_tries: u32, test_fn: F) -> bool
    where
        F: Fn() -> bool,
    {
        for _ in 0..n_tries {
            if test_fn() {
                return true;
            }
        }
        return false;
    }

    #[test]
    fn view_enum_test_gibbs() {
        fn test_fn() -> bool {
            let err = view_enum_test(4, 1, 1, 5_000, RowAssignAlg::Gibbs);
            println!("Error: {}", err);
            err < 0.01
        }
        assert!(flaky_test_passes(N_TRIES, test_fn));
    }

    #[test]
    #[ignore] // as of 9/28/18, this test fails
    fn view_enum_test_finite_cpu() {
        fn test_fn() -> bool {
            let err = view_enum_test(4, 1, 1, 5_000, RowAssignAlg::FiniteCpu);
            println!("Error: {}", err);
            err < 0.01
        }
        assert!(flaky_test_passes(N_TRIES, test_fn));
    }

    #[test]
    fn view_enum_test_slice() {
        fn test_fn() -> bool {
            let err = view_enum_test(4, 1, 1, 5_000, RowAssignAlg::Slice);
            println!("Error: {}", err);
            err < 0.01
        }
        assert!(flaky_test_passes(N_TRIES, test_fn));
    }
}

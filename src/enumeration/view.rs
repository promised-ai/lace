//! View Enumeration test
extern crate rand;

use std::collections::BTreeMap;

use self::rand::Rng;

use cc::assignment::lcrp;
use cc::transition::ViewTransition;
use cc::{
    AssignmentBuilder, ColModel, Feature, RowAssignAlg, View, ViewBuilder,
};
use enumeration::{build_features, normalize_assignment, partition_to_ix};
use misc::{logsumexp, Partition};

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

        let asgn = AssignmentBuilder::from_vec(z)
            .with_alpha(alpha)
            .build(&mut rng)
            .unwrap();

        let ln_pz = lcrp(n, &asgn.counts, alpha);

        let view: View = ViewBuilder::from_assignment(asgn)
            .with_features(features.clone())
            .build(&mut rng);

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
            .build(&mut rng)
            .unwrap();
        let mut view = ViewBuilder::from_assignment(asgn)
            .with_features(features.clone())
            .build(&mut rng);
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
    // TODO: Move enumeration test to integration tests
    #[test]
    fn view_enum_test_gibbs() {
        let err = view_enum_test(4, 1, 1, 5_000, RowAssignAlg::Gibbs);
        println!("Error: {}", err);
        assert!(err < 0.05);
    }

    #[test]
    #[ignore] // as of 9/28/18, this test fails
    fn view_enum_test_finite_cpu() {
        let err = view_enum_test(4, 1, 1, 5_000, RowAssignAlg::FiniteCpu);
        println!("Error: {}", err);
        assert!(err < 0.05);
    }

    #[test]
    fn view_enum_test_slice() {
        let err = view_enum_test(4, 1, 1, 5_000, RowAssignAlg::Slice);
        println!("Error: {}", err);
        assert!(err < 0.05);
    }
}

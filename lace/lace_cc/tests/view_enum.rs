//! View Enumeration test
//!
//! Tests the row assignment algorithms separately from the column algorithms.
mod enum_test;

use std::collections::BTreeMap;

use rand::Rng;

use enum_test::{
    build_features, normalize_assignment, partition_to_ix, Partition,
};
use lace_cc::alg::RowAssignAlg;
use lace_cc::feature::{ColModel, FType, Feature};
use lace_cc::transition::ViewTransition;
use lace_cc::view::{Builder, View};
use lace_stats::prior_process::Builder as PriorProcessBuilder;
use lace_stats::prior_process::{Dirichlet, PitmanYor, Process};
use lace_stats::rv::dist::{Beta, Gamma};
use lace_stats::rv::misc::logsumexp;

const N_TRIES: u32 = 5;

#[derive(Clone, Copy, Debug)]
pub enum ProcessType {
    Dirichlet,
    PitmanYor,
}

impl From<ProcessType> for Process {
    fn from(proc: ProcessType) -> Self {
        match proc {
            ProcessType::Dirichlet => Process::Dirichlet(Dirichlet {
                alpha: 1.0,
                prior: Gamma::default(),
            }),
            ProcessType::PitmanYor => Process::PitmanYor(PitmanYor {
                alpha: 1.2,
                d: 0.2,
                prior_alpha: Gamma::default(),
                prior_d: Beta::jeffreys(),
            }),
        }
    }
}

/// Compute the posterior of all assignments of the features under CRP(alpha)
///
/// NOTE: The rng is required, for calling AssignmentBuilder.build, but nothing
/// random should actually happen.
#[allow(clippy::ptr_arg)]
fn calc_partition_ln_posterior<R: Rng>(
    features: &Vec<ColModel>,
    proc_type: ProcessType,
    mut rng: &mut R,
) -> BTreeMap<u64, f64> {
    let n = features[0].len();
    let mut ln_posterior: BTreeMap<u64, f64> = BTreeMap::new();

    Partition::new(n).for_each(|z| {
        let ix = partition_to_ix(&z);

        // NOTE: We don't need seed control here because both alpha and the
        // assignment are set, but I'm setting the seed anyway in case the
        // assignment builder internals change
        let prior_process = PriorProcessBuilder::from_vec(z)
            .with_process(proc_type.into())
            .seed_from_rng(&mut rng)
            .build()
            .unwrap();

        // let ln_pz = lcrp(n, &prior_process.asgn.counts, alpha);
        let ln_pz = prior_process.ln_f_partition(&prior_process.asgn);

        let view: View = Builder::from_prior_process(prior_process)
            .features(features.clone())
            .seed_from_rng(&mut rng)
            .build();

        ln_posterior.insert(ix, view.score() + ln_pz);
    });

    ln_posterior
}

/// Normalize and exp the posterior computed in calc_partition_ln_posterior
fn norm_posterior(ln_posterior: &BTreeMap<u64, f64>) -> BTreeMap<u64, f64> {
    let logps: Vec<f64> = ln_posterior.values().copied().collect();
    let z = logsumexp(&logps);
    let mut normed: BTreeMap<u64, f64> = BTreeMap::new();
    for (key, lp) in ln_posterior {
        normed.insert(*key, (lp - z).exp());
    }
    normed
}

/// Compute the sum absolute error (in 0 to 1), for a View consisting of `n_cols`
/// continuous columns each with `n_rows` rows.
pub fn view_enum_test(
    n_rows: usize,
    n_cols: usize,
    n_runs: usize,
    n_iters: usize,
    ftype: FType,
    row_alg: RowAssignAlg,
    proc_type: ProcessType,
) -> f64 {
    let mut rng = rand::thread_rng();
    let features = build_features(n_rows, n_cols, ftype, &mut rng);
    let ln_posterior =
        calc_partition_ln_posterior(&features, proc_type, &mut rng);
    let posterior = norm_posterior(&ln_posterior);

    let transitions: Vec<ViewTransition> = vec![
        ViewTransition::RowAssignment(row_alg),
        ViewTransition::ComponentParams,
    ];

    let mut est_posterior: BTreeMap<u64, f64> = BTreeMap::new();
    let inc: f64 = ((n_runs * n_iters) as f64).recip();

    for _ in 0..n_runs {
        let prior_process = PriorProcessBuilder::new(n_rows)
            .with_process(proc_type.into())
            .seed_from_rng(&mut rng)
            .build()
            .unwrap();

        let mut view = Builder::from_prior_process(prior_process)
            .features(features.clone())
            .seed_from_rng(&mut rng)
            .build();

        for _ in 0..n_iters {
            view.update(10, &transitions, &mut rng);

            let normed = normalize_assignment(view.asgn().asgn.clone());
            let ix = partition_to_ix(&normed);

            if !posterior.contains_key(&ix) {
                panic!("invalid index!\n{:?}\n{:?}", view.asgn().asgn, normed);
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

fn flaky_test_passes<F>(n_tries: u32, test_fn: F) -> bool
where
    F: Fn() -> bool,
{
    for _ in 0..n_tries {
        if test_fn() {
            return true;
        }
    }
    false
}

// TODO: could remove $test name by using mods
macro_rules! view_enum_test {
    ($ftype: ident, $proctype: ident, $row_alg: ident) => {
        #[test]
        fn $row_alg() {
            fn test_fn() -> bool {
                let err = view_enum_test(
                    4,
                    1,
                    1,
                    5_000,
                    FType::$ftype,
                    RowAssignAlg::$row_alg,
                    ProcessType::$proctype,
                );
                eprintln!("err: {}", err);
                err < 0.01
            }
            assert!(flaky_test_passes(N_TRIES, test_fn));
        }
    };
    ($modname: ident, $ftype: ident, $proctype: ident, [$($row_alg: ident),+]) => {
        #[allow(non_snake_case)]
        mod $modname {
            use super::*;
            $(
                view_enum_test!($ftype, $proctype, $row_alg);
            )+
        }
    };
    ($(($modname: ident, $ftype: ident, $proctype: ident, $row_algs: tt)),+) => {
        $(
            view_enum_test!($modname, $ftype, $proctype, $row_algs);
        )+

    };
}

view_enum_test!(
    (
        ve_continuous_dp,
        Continuous,
        Dirichlet,
        [Gibbs, Slice, Sams]
    ),
    (
        ve_continuous_pyp,
        Continuous,
        PitmanYor,
        [Gibbs, Slice, Sams]
    ),
    (
        ve_categorical_dp,
        Categorical,
        Dirichlet,
        [Gibbs, Slice, Sams]
    ),
    (
        ve_categorical_pyp,
        Categorical,
        PitmanYor,
        [Gibbs, Slice, Sams]
    ),
    (ve_count_dp, Count, Dirichlet, [Gibbs, Slice, Sams]),
    (ve_count_pyp, Count, PitmanYor, [Gibbs, Slice, Sams])
);

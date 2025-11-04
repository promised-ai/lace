//! State Enumeration test
//!
//! Enumeration tests work by enumerating all the possible cross-categorization
//! bi-partitions, computing the data likelihood and partition prior, normalizing
//! and comparing against the empirical posterior from the sampler. None of the
//! prior parameters are re-sampled: rhe feature priors, and CRP alphas are static.
//! If the algorithm is correct, the estimated and true posterior should be very
//! close.
mod enum_test;

use std::collections::HashMap;

use itertools::Itertools;
use rand::Rng;
use rv::misc::LogSumExp;

use lace::cc::alg::{ColAssignAlg, RowAssignAlg};
use lace::cc::config::StateUpdateConfig;
use lace::cc::feature::{ColModel, FType, Feature};
use lace::cc::state::State;
use lace::cc::transition::StateTransition;
use lace::cc::view::{Builder, View};
use lace::stats::prior_process::Builder as PriorProcessBuilder;
use lace::stats::prior_process::{
    Dirichlet, PitmanYor, PriorProcessT, PriorProcessType, Process,
};
use rv::dist::{Beta, Gamma};

use enum_test::{
    build_features, normalize_assignment, partition_to_ix, Partition,
};

type StateIndex = (u64, Vec<u64>);

#[derive(Debug, Clone)]
struct StatePartition {
    col_partition: Vec<usize>,
    row_partitions: Vec<Vec<usize>>,
}

impl StatePartition {
    /// Convert the partitions the a compact, standardized, hashable index.
    fn get_index(&self) -> StateIndex {
        let col_ix = {
            let z = normalize_assignment(self.col_partition.clone());
            partition_to_ix(&z)
        };
        let row_ixs: Vec<u64> = self
            .row_partitions
            .iter()
            .map(|zr| {
                let z = normalize_assignment(zr.clone());
                partition_to_ix(&z)
            })
            .collect();
        (col_ix, row_ixs)
    }
}

fn enumerate_state_partitions(
    n_rows: usize,
    n_cols: usize,
) -> Vec<StatePartition> {
    let mut state_parts: Vec<StatePartition> = vec![];
    Partition::new(n_cols).for_each(|zc| {
        let k = zc
            .iter()
            .fold(0, |max, &zi| if max < zi { zi } else { max });

        (0..=k)
            .map(|_| Partition::new(n_rows))
            .multi_cartesian_product()
            .for_each(|zr| {
                let state_part = StatePartition {
                    col_partition: zc.clone(),
                    row_partitions: zr,
                };
                state_parts.push(state_part);
            });
    });
    state_parts
}

/// Generates a enumeration-test-ready State given a partition
fn state_from_partition<R: Rng>(
    partition: &StatePartition,
    mut features: Vec<ColModel>,
    mut rng: &mut R,
) -> State {
    let mut views: Vec<View> = partition
        .row_partitions
        .iter()
        .map(|zr| {
            let process = Process::Dirichlet(Dirichlet {
                alpha: 1.0,
                alpha_prior: Gamma::default(),
            });

            // NOTE: We don't need seed control here because both alpha and the
            // assignment are set, but I'm setting the seed anyway in case the
            // assignment builder internals change
            let view_prior_process = PriorProcessBuilder::from_vec(zr.clone())
                .with_process(process.clone())
                .seed_from_rng(&mut rng)
                .build()
                .unwrap();
            Builder::from_prior_process(view_prior_process)
                .seed_from_rng(&mut rng)
                .build()
        })
        .collect();

    partition
        .col_partition
        .iter()
        .zip(features.drain(..))
        .for_each(|(&zi, ftr)| views[zi].insert_feature(ftr, &mut rng));

    let state_prior_process =
        PriorProcessBuilder::from_vec(partition.col_partition.clone())
            .with_process(Process::Dirichlet(Dirichlet {
                alpha: 1.0,
                alpha_prior: Gamma::default(),
            }))
            .seed_from_rng(&mut rng)
            .build()
            .unwrap();

    State::new(views, state_prior_process)
}

fn emit_process(proc_type: PriorProcessType) -> Process {
    match proc_type {
        PriorProcessType::Dirichlet => Process::Dirichlet(Dirichlet {
            alpha: 1.0,
            alpha_prior: Gamma::default(),
        }),
        PriorProcessType::PitmanYor => Process::PitmanYor(PitmanYor {
            alpha: 1.2,
            d: 0.2,
            alpha_prior: Gamma::default(),
            d_prior: Beta::jeffreys(),
        }),
    }
}

/// Generates a random start state from the prior, with default values chosen for the
/// feature priors, and all CRP alphas set to 1.0.
fn gen_start_state<R: Rng>(
    mut features: Vec<ColModel>,
    proc_type: PriorProcessType,
    mut rng: &mut R,
) -> State {
    let n_cols = features.len();
    let n_rows = features[0].len();

    let process = emit_process(proc_type);

    let state_prior_process = PriorProcessBuilder::new(n_cols)
        .with_process(process)
        .seed_from_rng(&mut rng)
        .build()
        .unwrap();

    let mut views: Vec<View> = (0..state_prior_process.asgn.n_cats)
        .map(|_| {
            let view_prior_process = PriorProcessBuilder::new(n_rows)
                .with_process(state_prior_process.process.clone())
                .seed_from_rng(&mut rng)
                .build()
                .unwrap();
            Builder::from_prior_process(view_prior_process).build()
        })
        .collect();

    state_prior_process
        .asgn
        .iter()
        .zip(features.drain(..))
        .for_each(|(&zi, ftr)| views[zi].insert_feature(ftr, &mut rng));

    State::new(views, state_prior_process)
}

fn calc_state_ln_posterior<R: Rng>(
    features: Vec<ColModel>,
    state_process: &Process,
    view_process: &Process,
    mut rng: &mut R,
) -> HashMap<StateIndex, f64> {
    let n_cols = features.len();
    let n_rows = features[0].len();

    let mut ln_posterior: HashMap<StateIndex, f64> = HashMap::new();

    enumerate_state_partitions(n_rows, n_cols)
        .iter()
        .for_each(|part| {
            let state = state_from_partition(part, features.clone(), &mut rng);
            let mut score = state_process.ln_f_partition(state.asgn());
            for view in state.views {
                score += view_process.ln_f_partition(view.asgn());
                for ftr in view.ftrs.values() {
                    score += ftr.score();
                }
            }
            ln_posterior.insert(part.get_index(), score);
        });
    let norm = ln_posterior.values().logsumexp();

    ln_posterior
        .values_mut()
        .for_each(|v| *v = (*v - norm).exp());
    ln_posterior
}

/// Extract the index from a State
fn extract_state_index(state: &State) -> StateIndex {
    let normed = normalize_assignment(state.asgn().asgn.clone());
    let col_ix: u64 = partition_to_ix(&normed);
    let row_ixs: Vec<u64> = state
        .views
        .iter()
        .map(|v| {
            let zn = normalize_assignment(v.asgn().asgn.clone());
            partition_to_ix(&zn)
        })
        .collect();
    (col_ix, row_ixs)
}

/// Do the state enumeration test
///
/// # Arguments
/// - n_rows: the number of rows in the table
/// - n_cols: the number of columns in the table
/// - n_runs: the number of restarts
/// - n_iters: the number of MCMC iterations for each run
/// - row_alg: the row assignment algorithm to test
/// - col_alg: the column assignment algorithm to test
// TODO: Change arguments to a struct that implements Default and the rng
#[allow(clippy::too_many_arguments)]
pub fn state_enum_test<R: Rng>(
    n_rows: usize,
    n_cols: usize,
    n_runs: usize,
    n_iters: usize,
    row_alg: RowAssignAlg,
    col_alg: ColAssignAlg,
    ftype: FType,
    proc_type: PriorProcessType,
    mut rng: &mut R,
) -> f64 {
    let features = build_features(n_rows, n_cols, ftype, &mut rng);
    let mut est_posterior: HashMap<StateIndex, f64> = HashMap::new();
    let update_config = StateUpdateConfig {
        n_iters: 1,
        transitions: vec![
            StateTransition::ColumnAssignment(col_alg),
            StateTransition::RowAssignment(row_alg),
            StateTransition::ComponentParams,
        ],
    };

    let inc: f64 = ((n_runs * n_iters) as f64).recip();

    for _ in 0..n_runs {
        let mut state = gen_start_state(features.clone(), proc_type, &mut rng);

        for _ in 0..n_iters {
            state.update(update_config.clone(), &mut rng);

            let ix = extract_state_index(&state);
            *est_posterior.entry(ix).or_insert(0.0) += inc;
        }
    }

    let process = emit_process(proc_type);

    let posterior =
        calc_state_ln_posterior(features, &process, &process, &mut rng);

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
    use lace::utils::numbers::ccnum;

    const N_TRIES: u32 = 10;

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
    macro_rules! state_enum_test {
        ($test_name: ident, $ftype: ident, $row_alg: ident, $col_alg: ident) => {
            mod $test_name {
                use super::*;

                #[test]
                fn dirichlet() {
                    fn test_fn() -> bool {
                        let mut rng = rand::rng();
                        let err = state_enum_test(
                            3,
                            3,
                            1,
                            10_000,
                            RowAssignAlg::$row_alg,
                            ColAssignAlg::$col_alg,
                            FType::$ftype,
                            PriorProcessType::Dirichlet,
                            &mut rng,
                        );
                        eprintln!("err: {err}");
                        err < 0.01
                    }
                    assert!(flaky_test_passes(N_TRIES, test_fn));
                }

                #[test]
                fn pitman_yor() {
                    fn test_fn() -> bool {
                        let mut rng = rand::rng();
                        let err = state_enum_test(
                            3,
                            3,
                            1,
                            10_000,
                            RowAssignAlg::$row_alg,
                            ColAssignAlg::$col_alg,
                            FType::$ftype,
                            PriorProcessType::PitmanYor,
                            &mut rng,
                        );
                        eprintln!("err: {err}");
                        err < 0.01
                    }
                    assert!(flaky_test_passes(N_TRIES, test_fn));
                }
            }
        };
        ($(($fn_name: ident, $ftype: ident, $row_alg: ident, $col_alg: ident)),+) => {
            $(
                state_enum_test!($fn_name, $ftype, $row_alg, $col_alg);
            )+

        };
    }

    #[test]
    fn enum_state_partitions_should_produce_correct_number() {
        assert_eq!(enumerate_state_partitions(3, 3).len(), 205);
        assert_eq!(
            enumerate_state_partitions(3, 4).len(),
            ccnum(3, 4) as usize
        );
    }

    #[test]
    fn ln_posterior_length() {
        let process = emit_process(PriorProcessType::Dirichlet);
        let mut rng = rand::rng();
        let ftrs = build_features(3, 3, FType::Continuous, &mut rng);
        let posterior =
            calc_state_ln_posterior(ftrs, &process, &process, &mut rng);
        assert_eq!(posterior.len(), 205)
    }

    // XXX: Finite CPU algorithm fails this because it is an approximate
    // algorithm, so we do not include it here.
    state_enum_test!(
        // Continuous
        (
            state_enum_test_continuous_gibbs_gibbs,
            Continuous,
            Gibbs,
            Gibbs
        ),
        (
            state_enum_test_continuous_slice_slice,
            Continuous,
            Slice,
            Slice
        ),
        (
            state_enum_test_continuous_gibbs_slice,
            Continuous,
            Gibbs,
            Slice
        ),
        (
            state_enum_test_continuous_slice_gibbs,
            Continuous,
            Slice,
            Gibbs
        ),
        (
            state_enum_test_continuous_sams_slice,
            Continuous,
            Sams,
            Slice
        ),
        (
            state_enum_test_continuous_sams_gibb,
            Continuous,
            Sams,
            Gibbs
        ),
        // Categorical
        (
            state_enum_test_categorical_gibbs_gibbs,
            Categorical,
            Gibbs,
            Gibbs
        ),
        (
            state_enum_test_categorical_slice_slice,
            Categorical,
            Slice,
            Slice
        ),
        (
            state_enum_test_categorical_gibbs_slice,
            Categorical,
            Gibbs,
            Slice
        ),
        (
            state_enum_test_categorical_slice_gibbs,
            Categorical,
            Slice,
            Gibbs
        ),
        (
            state_enum_test_categorical_sams_slice,
            Categorical,
            Sams,
            Slice
        ),
        (
            state_enum_test_categorical_sama_gibbs,
            Categorical,
            Sams,
            Gibbs
        ),
        // Count
        (state_enum_test_count_gibbs_gibbs, Count, Gibbs, Gibbs),
        (state_enum_test_count_slice_slice, Count, Slice, Slice),
        (state_enum_test_count_gibbs_slice, Count, Gibbs, Slice),
        (state_enum_test_count_slice_gibbs, Count, Slice, Gibbs),
        (state_enum_test_count_sams_slice, Count, Sams, Slice),
        (state_enum_test_count_sams_gibbs, Count, Sams, Gibbs)
    );
}

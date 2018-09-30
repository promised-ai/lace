//! State Enumeration test
extern crate itertools;
extern crate rand;
extern crate rv;

use std::collections::HashMap;

use self::itertools::Itertools;
use self::rand::Rng;
use self::rv::dist::Gamma;

use cc::assignment::lcrp;
use cc::config::StateUpdateConfig;
use cc::transition::StateTransition;
use cc::{
    AssignmentBuilder, ColAssignAlg, ColModel, Feature, RowAssignAlg, State,
    View, ViewBuilder,
};
use enumeration::{build_features, normalize_assignment, partition_to_ix};
use misc::{logsumexp, Partition};

type StateIndex = (u64, Vec<u64>);

#[derive(Debug, Clone)]
struct StatePartition {
    col_partition: Vec<usize>,
    row_partitions: Vec<Vec<usize>>,
}

impl StatePartition {
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
            }).collect();
        (col_ix, row_ixs)
    }
}

fn enumerate_state_partitions(
    nrows: usize,
    ncols: usize,
) -> Vec<StatePartition> {
    let mut state_parts: Vec<StatePartition> = vec![];
    Partition::new(ncols).for_each(|zc| {
        let k = zc
            .iter()
            .fold(0, |max, &zi| if max < zi { zi } else { max });

        (0..=k)
            .map(|_| Partition::new(nrows))
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

fn state_from_partition<R: Rng>(
    partition: &StatePartition,
    mut features: Vec<ColModel>,
    mut rng: &mut R,
) -> State {
    let asgn = AssignmentBuilder::from_vec(partition.col_partition.clone())
        .with_alpha(1.0)
        .build(&mut rng)
        .unwrap();

    let mut views: Vec<View> = partition
        .row_partitions
        .iter()
        .map(|zr| {
            let asgn = AssignmentBuilder::from_vec(zr.clone())
                .with_alpha(1.0)
                .build(&mut rng)
                .unwrap();
            ViewBuilder::from_assignment(asgn).build(&mut rng)
        }).collect();

    partition
        .col_partition
        .iter()
        .zip(features.drain(..))
        .for_each(|(&zi, ftr)| views[zi].insert_feature(ftr, &mut rng));

    State::new(views, asgn, Gamma::new(1.0, 1.0).unwrap())
}

fn gen_start_state<R: Rng>(
    mut features: Vec<ColModel>,
    mut rng: &mut R,
) -> State {
    let ncols = features.len();
    let nrows = features[0].len();
    let asgn = AssignmentBuilder::new(ncols)
        .with_alpha(1.0)
        .build(&mut rng)
        .unwrap();

    let mut views: Vec<View> = (0..asgn.ncats)
        .map(|zr| {
            let asgn = AssignmentBuilder::new(nrows)
                .with_alpha(1.0)
                .build(&mut rng)
                .unwrap();
            ViewBuilder::from_assignment(asgn).build(&mut rng)
        }).collect();

    asgn.iter()
        .zip(features.drain(..))
        .for_each(|(&zi, ftr)| views[zi].insert_feature(ftr, &mut rng));

    State::new(views, asgn, Gamma::new(1.0, 1.0).unwrap())
}

fn calc_state_ln_posterior<R: Rng>(
    features: Vec<ColModel>,
    mut rng: &mut R,
) -> HashMap<StateIndex, f64> {
    let ncols = features.len();
    let nrows = features[0].len();

    let mut ln_posterior: HashMap<StateIndex, f64> = HashMap::new();

    enumerate_state_partitions(nrows, ncols)
        .iter()
        .for_each(|part| {
            let state = state_from_partition(part, features.clone(), &mut rng);
            let mut score = lcrp(state.ncols(), &state.asgn.counts, 1.0);
            for view in state.views {
                score += lcrp(view.nrows(), &view.asgn.counts, 1.0);
                for ftr in view.ftrs.values() {
                    score += ftr.score();
                }
            }
            ln_posterior.insert(part.get_index(), score);
        });
    let norm = {
        let scores: Vec<f64> = ln_posterior.values().map(|&v| v).collect();
        logsumexp(&scores)
    };

    ln_posterior
        .values_mut()
        .for_each(|v| *v = (*v - norm).exp());
    ln_posterior
}

fn extract_state_index(state: &State) -> StateIndex {
    let normed = normalize_assignment(state.asgn.asgn.clone());
    let col_ix: u64 = partition_to_ix(&normed);
    let row_ixs: Vec<u64> = state
        .views
        .iter()
        .map(|ref v| {
            let zn = normalize_assignment(v.asgn.asgn.clone());
            partition_to_ix(&zn)
        }).collect();
    (col_ix, row_ixs)
}

pub fn state_enum_test<R: Rng>(
    nrows: usize,
    ncols: usize,
    n_runs: usize,
    n_iters: usize,
    row_alg: RowAssignAlg,
    col_alg: ColAssignAlg,
    mut rng: &mut R,
) -> f64 {
    let features = build_features(nrows, ncols, &mut rng);
    let mut est_posterior: HashMap<StateIndex, f64> = HashMap::new();
    let update_config = StateUpdateConfig::new()
        .with_iters(1)
        .with_col_alg(col_alg)
        .with_row_alg(row_alg)
        .with_transitions(vec![
            StateTransition::ColumnAssignment,
            StateTransition::RowAssignment,
            StateTransition::ComponentParams,
        ]);

    let inc: f64 = ((n_runs * n_iters) as f64).recip();

    for _ in 0..n_runs {
        let mut state = gen_start_state(features.clone(), &mut rng);

        // alphas should start out at 1.0
        assert!((state.asgn.alpha - 1.0).abs() < 1E-16);
        assert!(
            state
                .views
                .iter()
                .all(|v| (v.asgn.alpha - 1.0).abs() < 1E-16)
        );

        for _ in 0..n_iters {
            state.update(update_config.clone(), &mut rng);

            // all alphas should be 1.0
            assert!((state.asgn.alpha - 1.0).abs() < 1E-16);
            assert!(
                state
                    .views
                    .iter()
                    .all(|v| (v.asgn.alpha - 1.0).abs() < 1E-16)
            );

            let ix = extract_state_index(&state);
            *est_posterior.entry(ix).or_insert(0.0) += inc;
        }
    }

    let posterior = calc_state_ln_posterior(features, &mut rng);

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
    use misc::ccnum;

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
        let mut rng = rand::thread_rng();
        let ftrs = build_features(3, 3, &mut rng);
        let posterior = calc_state_ln_posterior(ftrs, &mut rng);
        assert_eq!(posterior.len(), 205)
    }

    #[test]
    fn state_enum_test_slice_slice() {
        let mut rng = rand::thread_rng();
        let err = state_enum_test(
            3,
            3,
            1,
            10000,
            RowAssignAlg::Slice,
            ColAssignAlg::Slice,
            &mut rng,
        );
        println!("Error: {}", err);
        assert!(err < 0.05);
    }

    // FIXME: Doesn't work w/ gibbs becausue new views always draw alpha from
    // the prior.
    #[test]
    #[ignore]
    fn state_enum_test_gibbs_gibbs() {
        let mut rng = rand::thread_rng();
        let err = state_enum_test(
            3,
            3,
            1,
            10000,
            RowAssignAlg::Gibbs,
            ColAssignAlg::Gibbs,
            &mut rng,
        );
        println!("Error: {}", err);
        assert!(err < 0.05);
    }
}

use braid::cc::config::StateUpdateConfig;
use braid::cc::{ColAssignAlg, RowAssignAlg};
use braid::cc::{ColModel, Column, FeatureData, State};
use braid_data::SparseContainer;
use braid_stats::prior::ng::NgHyper;
use rand::Rng;
use rv::dist::{Gamma, Gaussian, NormalInvGamma};
use rv::traits::Rv;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let hyper = NgHyper::default();
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = SparseContainer::from(data_vec);
    let prior = NormalInvGamma::new_unchecked(0.0, 1.0, 40.0, 40.0);

    let ftr = Column::new(id, data, prior, hyper);
    ColModel::Continuous(ftr)
}

fn gen_all_gauss_state<R: Rng>(
    nrows: usize,
    ncols: usize,
    mut rng: &mut R,
) -> State {
    let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
    for i in 0..ncols {
        ftrs.push(gen_col(i, nrows, &mut rng));
    }
    State::from_prior(
        ftrs,
        Gamma::new(1.0, 1.0).unwrap().into(),
        Gamma::new(1.0, 1.0).unwrap().into(),
        &mut rng,
    )
}

#[test]
fn smoke() {
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(10, 2, &mut rng);

    assert_eq!(state.nrows(), 10);
    assert_eq!(state.ncols(), 2);

    let config = StateUpdateConfig {
        n_iters: 100,
        ..Default::default()
    };
    state.update(config, &mut rng);
}

#[test]
fn drop_data_should_remove_data_from_all_fatures() {
    let nrows = 10;
    let ncols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert_eq!(ftr.data.len(), nrows),
            _ => panic!("Unexpected column type"),
        }
    }

    state.drop_data();

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }
}

#[test]
fn take_data_should_remove_data_from_all_fatures() {
    let nrows = 10;
    let ncols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert_eq!(ftr.data.len(), nrows),
            _ => panic!("Unexpected column type"),
        }
    }

    let data = state.take_data();
    assert_eq!(data.len(), ncols);
    for id in 0..ncols {
        assert!(data.contains_key(&id));
    }

    for data_col in data.values() {
        match data_col {
            &FeatureData::Continuous(ref xs) => assert_eq!(xs.len(), nrows),
            _ => panic!("Unexpected data types"),
        }
    }

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }
}

#[test]
fn repop_data_should_return_the_data_to_all_fatures() {
    let nrows = 10;
    let ncols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert_eq!(ftr.data.len(), nrows),
            _ => panic!("Unexpected column type"),
        }
    }

    let data = state.take_data();

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }

    // should panic if something goes wrong
    state.repop_data(data);

    assert_eq!(state.ncols(), ncols);
    assert_eq!(state.nrows(), nrows);

    for id in 0..ncols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert_eq!(ftr.data.len(), nrows),
            _ => panic!("Unexpected column type"),
        }
    }
}

#[test]
fn insert_new_features_should_work() {
    let nrows = 10;
    let ncols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    let ftrs: Vec<ColModel> = (0..3)
        .map(|i| gen_col(i + ncols, nrows, &mut rng))
        .collect();

    assert_eq!(state.ncols(), 5);
    state.insert_new_features(ftrs, &mut rng);
    assert_eq!(state.ncols(), 8);
}

fn two_part_runner(
    first_algs: (RowAssignAlg, ColAssignAlg),
    second_algs: (RowAssignAlg, ColAssignAlg),
    mut rng: &mut impl Rng,
) {
    use braid::cc::StateTransition;
    let nrows = 100;
    let ncols = 20;

    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    let update_config_1 = StateUpdateConfig {
        n_iters: 50,
        transitions: vec![
            StateTransition::ColumnAssignment(first_algs.1),
            StateTransition::StateAlpha,
            StateTransition::RowAssignment(first_algs.0),
            StateTransition::ViewAlphas,
            StateTransition::FeaturePriors,
        ],
        ..Default::default()
    };

    state.update(update_config_1, &mut rng);

    let update_config_2 = StateUpdateConfig {
        n_iters: 50,
        transitions: vec![
            StateTransition::ColumnAssignment(second_algs.1),
            StateTransition::StateAlpha,
            StateTransition::RowAssignment(second_algs.0),
            StateTransition::ViewAlphas,
            StateTransition::FeaturePriors,
        ],
        ..Default::default()
    };

    state.update(update_config_2, &mut rng);
}

#[test]
fn run_slice_row_after_finite() {
    let mut rng = rand::thread_rng();
    two_part_runner(
        (RowAssignAlg::FiniteCpu, ColAssignAlg::FiniteCpu),
        (RowAssignAlg::Slice, ColAssignAlg::FiniteCpu),
        &mut rng,
    );
}

#[test]
fn run_slice_col_after_gibbs() {
    let mut rng = rand::thread_rng();
    two_part_runner(
        (RowAssignAlg::FiniteCpu, ColAssignAlg::Gibbs),
        (RowAssignAlg::FiniteCpu, ColAssignAlg::Slice),
        &mut rng,
    );
}

#[test]
fn run_slice_row_after_gibbs() {
    // 2018-12-20 This used to cause subtract w/ overflow or out of bounds error
    // because the Slice sampler wasn't cleaning up the weights that the Gibbs
    // sampler was neglecting.
    let mut rng = rand::thread_rng();
    two_part_runner(
        (RowAssignAlg::Gibbs, ColAssignAlg::FiniteCpu),
        (RowAssignAlg::Slice, ColAssignAlg::FiniteCpu),
        &mut rng,
    );
}

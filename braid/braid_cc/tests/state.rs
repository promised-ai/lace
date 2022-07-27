use braid_cc::alg::{ColAssignAlg, RowAssignAlg};
use braid_cc::config::StateUpdateConfig;
use braid_cc::feature::{ColModel, Column};
use braid_cc::state::State;
use braid_data::{FeatureData, SparseContainer};
use braid_stats::prior::nix::NixHyper;
use rand::Rng;
use rv::dist::{Gamma, Gaussian, NormalInvChiSquared};
use rv::traits::Rv;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let hyper = NixHyper::default();
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = SparseContainer::from(data_vec);
    let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 40.0, 40.0);

    let ftr = Column::new(id, data, prior, hyper);
    ColModel::Continuous(ftr)
}

fn gen_all_gauss_state<R: Rng>(
    n_rows: usize,
    n_cols: usize,
    mut rng: &mut R,
) -> State {
    let mut ftrs: Vec<ColModel> = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        ftrs.push(gen_col(i, n_rows, &mut rng));
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

    assert_eq!(state.n_rows(), 10);
    assert_eq!(state.n_cols(), 2);

    let config = StateUpdateConfig {
        n_iters: 100,
        ..Default::default()
    };
    state.update(config, &mut rng);
}

#[test]
fn drop_data_should_remove_data_from_all_fatures() {
    let n_rows = 10;
    let n_cols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(n_rows, n_cols, &mut rng);

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => {
                assert_eq!(ftr.data.len(), n_rows)
            }
            _ => panic!("Unexpected column type"),
        }
    }

    state.drop_data();

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }
}

#[test]
fn take_data_should_remove_data_from_all_fatures() {
    let n_rows = 10;
    let n_cols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(n_rows, n_cols, &mut rng);

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => {
                assert_eq!(ftr.data.len(), n_rows)
            }
            _ => panic!("Unexpected column type"),
        }
    }

    let data = state.take_data();
    assert_eq!(data.len(), n_cols);
    for id in 0..n_cols {
        assert!(data.contains_key(&id));
    }

    for data_col in data.values() {
        match data_col {
            &FeatureData::Continuous(ref xs) => assert_eq!(xs.len(), n_rows),
            _ => panic!("Unexpected data types"),
        }
    }

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }
}

#[test]
fn repop_data_should_return_the_data_to_all_fatures() {
    let n_rows = 10;
    let n_cols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(n_rows, n_cols, &mut rng);

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => {
                assert_eq!(ftr.data.len(), n_rows)
            }
            _ => panic!("Unexpected column type"),
        }
    }

    let data = state.take_data();

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }

    // should panic if something goes wrong
    state.repop_data(data);

    assert_eq!(state.n_cols(), n_cols);
    assert_eq!(state.n_rows(), n_rows);

    for id in 0..n_cols {
        match state.feature(id) {
            &ColModel::Continuous(ref ftr) => {
                assert_eq!(ftr.data.len(), n_rows)
            }
            _ => panic!("Unexpected column type"),
        }
    }
}

#[test]
fn insert_new_features_should_work() {
    let n_rows = 10;
    let n_cols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(n_rows, n_cols, &mut rng);

    let ftrs: Vec<ColModel> = (0..3)
        .map(|i| gen_col(i + n_cols, n_rows, &mut rng))
        .collect();

    assert_eq!(state.n_cols(), 5);
    state.insert_new_features(ftrs, &mut rng);
    assert_eq!(state.n_cols(), 8);
}

fn two_part_runner(
    first_algs: (RowAssignAlg, ColAssignAlg),
    second_algs: (RowAssignAlg, ColAssignAlg),
    mut rng: &mut impl Rng,
) {
    use braid_cc::transition::StateTransition;
    let n_rows = 100;
    let n_cols = 20;

    let mut state = gen_all_gauss_state(n_rows, n_cols, &mut rng);

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

#[test]
fn del_col_front() {
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(10, 5, &mut rng);

    assert_eq!(state.n_cols(), 5);

    let xs: Vec<f64> = (1..5)
        .map(|ix| state.datum(0, ix).to_f64_opt().unwrap())
        .collect();

    state.del_col(0, &mut rng);

    let ys: Vec<f64> = (0..4)
        .map(|ix| state.datum(0, ix).to_f64_opt().unwrap())
        .collect();

    assert_eq!(xs, ys);
}

#[test]
fn del_col_mid() {
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(10, 5, &mut rng);

    assert_eq!(state.n_cols(), 5);

    let xs = {
        let mut xs_before: Vec<f64> = (0..2)
            .map(|ix| state.datum(0, ix).to_f64_opt().unwrap())
            .collect();

        let xs_after: Vec<f64> = (3..5)
            .map(|ix| state.datum(0, ix).to_f64_opt().unwrap())
            .collect();

        xs_before.extend(xs_after);
        xs_before
    };

    state.del_col(2, &mut rng);

    let ys: Vec<f64> = (0..4)
        .map(|ix| state.datum(0, ix).to_f64_opt().unwrap())
        .collect();

    assert_eq!(xs, ys);
}

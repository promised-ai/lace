#[macro_use]
extern crate approx;
extern crate braid;
extern crate braid_stats;
extern crate rand;
extern crate rv;
extern crate serde_yaml;

use self::braid_stats::prior::{Ng, NigHyper};
use self::rand::Rng;
use self::rv::dist::{Gamma, Gaussian};
use self::rv::traits::Rv;

use braid::cc::alg::{ColAssignAlg, RowAssignAlg};
use braid::cc::config::StateUpdateConfig;
use braid::cc::container::FeatureData;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::DataContainer;
use braid::cc::State;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let hyper = NigHyper::default();
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = Ng::new(0.0, 1.0, 4.0, 4.0, hyper);

    let ftr = Column::new(id, data, prior);
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
        Gamma::new(1.0, 1.0).unwrap(),
        Gamma::new(1.0, 1.0).unwrap(),
        &mut rng,
    )
}

#[test]
fn smoke() {
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(10, 2, &mut rng);

    assert_eq!(state.nrows(), 10);
    assert_eq!(state.ncols(), 2);

    let config = StateUpdateConfig::new().with_iters(100);
    state.update(config, &mut rng);
}

#[test]
fn drop_data_should_remove_data_from_all_fatures() {
    let nrows = 10;
    let ncols = 5;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    for id in 0..ncols {
        match state.get_feature(id) {
            &ColModel::Continuous(ref ftr) => assert_eq!(ftr.data.len(), nrows),
            _ => panic!("Unexpected column type"),
        }
    }

    state.drop_data();

    for id in 0..ncols {
        match state.get_feature(id) {
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
        match state.get_feature(id) {
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
        match state.get_feature(id) {
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
        match state.get_feature(id) {
            &ColModel::Continuous(ref ftr) => assert_eq!(ftr.data.len(), nrows),
            _ => panic!("Unexpected column type"),
        }
    }

    let data = state.take_data();

    for id in 0..ncols {
        match state.get_feature(id) {
            &ColModel::Continuous(ref ftr) => assert!(ftr.data.is_empty()),
            _ => panic!("Unexpected column type"),
        }
    }

    assert!(state.repop_data(data).is_ok());
    assert_eq!(state.ncols(), ncols);
    assert_eq!(state.nrows(), nrows);

    for id in 0..ncols {
        match state.get_feature(id) {
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
    state
        .insert_new_features(ftrs, &mut rng)
        .expect("insert new feature failed");
    assert_eq!(state.ncols(), 8);
}

fn two_part_runner(
    first_algs: (RowAssignAlg, ColAssignAlg),
    second_algs: (RowAssignAlg, ColAssignAlg),
    mut rng: &mut impl Rng,
) {
    let nrows = 100;
    let ncols = 20;
    let n_iters = 50;

    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    let update_config_finite = StateUpdateConfig::new()
        .with_iters(n_iters)
        .with_row_alg(first_algs.0)
        .with_col_alg(first_algs.1);

    state.update(update_config_finite, &mut rng);

    let update_config_slice_row = StateUpdateConfig::new()
        .with_iters(50)
        .with_row_alg(second_algs.0)
        .with_col_alg(second_algs.1);

    state.update(update_config_slice_row, &mut rng);
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
fn append_row() {
    use braid::cc::{AppendRowsData, Datum};

    let nrows = 10;
    let ncols = 4;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    assert_eq!(state.nrows(), 10);

    let new_row = vec![
        AppendRowsData::new(2, vec![Datum::Missing]),
        AppendRowsData::new(1, vec![Datum::Missing]),
        AppendRowsData::new(3, vec![Datum::Continuous(4.4)]),
        AppendRowsData::new(0, vec![Datum::Continuous(1.1)]),
    ];

    state.append_rows(new_row, &mut rng);

    assert_eq!(state.nrows(), 11);

    let y_0 = state.get_datum(10, 0).as_f64().unwrap();
    let y_1 = state.get_datum(10, 1);
    let y_2 = state.get_datum(10, 2);
    let y_3 = state.get_datum(10, 3).as_f64().unwrap();

    assert_relative_eq!(y_0, 1.1, epsilon = 1E-10);
    assert_relative_eq!(y_3, 4.4, epsilon = 1E-10);
    assert_eq!(y_1, Datum::Missing);
    assert_eq!(y_2, Datum::Missing);
}

#[test]
fn append_rows() {
    use braid::cc::{AppendRowsData, Datum};

    let nrows = 10;
    let ncols = 4;
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(nrows, ncols, &mut rng);

    assert_eq!(state.nrows(), 10);

    let x_0 =
        AppendRowsData::new(0, vec![Datum::Continuous(1.1), Datum::Missing]);
    let x_1 =
        AppendRowsData::new(1, vec![Datum::Missing, Datum::Continuous(3.3)]);
    let x_2 =
        AppendRowsData::new(2, vec![Datum::Missing, Datum::Continuous(2.2)]);
    let x_3 =
        AppendRowsData::new(3, vec![Datum::Continuous(4.4), Datum::Missing]);

    let new_row = vec![x_0, x_1, x_2, x_3];

    state.append_rows(new_row, &mut rng);

    assert_eq!(state.nrows(), 12);

    let y_00 = state.get_datum(10, 0).as_f64().unwrap();
    let y_01 = state.get_datum(10, 1);
    let y_02 = state.get_datum(10, 2);
    let y_03 = state.get_datum(10, 3).as_f64().unwrap();

    assert_relative_eq!(y_00, 1.1, epsilon = 1E-10);
    assert_relative_eq!(y_03, 4.4, epsilon = 1E-10);
    assert_eq!(y_01, Datum::Missing);
    assert_eq!(y_02, Datum::Missing);

    let y_10 = state.get_datum(11, 0);
    let y_11 = state.get_datum(11, 1).as_f64().unwrap();
    let y_12 = state.get_datum(11, 2).as_f64().unwrap();
    let y_13 = state.get_datum(11, 3);

    assert_relative_eq!(y_11, 3.3, epsilon = 1E-10);
    assert_relative_eq!(y_12, 2.2, epsilon = 1E-10);
    assert_eq!(y_10, Datum::Missing);
    assert_eq!(y_13, Datum::Missing);
}

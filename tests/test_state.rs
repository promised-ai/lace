extern crate braid;
extern crate rand;
extern crate rv;
extern crate serde_yaml;

use self::rand::Rng;
use braid::cc::config::StateUpdateConfig;
use braid::cc::container::FeatureData;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::DataContainer;
use braid::cc::State;
use braid::dist::prior::ng::NigHyper;
use braid::dist::prior::Ng;

use self::rv::dist::{Gamma, Gaussian};
use self::rv::traits::Rv;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let hyper = NigHyper::default();
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = Ng::new(0.0, 1.0, 1.0, 1.0, hyper);

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

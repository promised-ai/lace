extern crate braid;
extern crate rand;
extern crate serde_yaml;

use self::rand::Rng;
use braid::cc::container::FeatureData;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::DataContainer;
use braid::cc::State;
use braid::dist::prior::nig::NigHyper;
use braid::dist::prior::NormalInverseGamma;
use braid::dist::traits::RandomVariate;
use braid::dist::Gaussian;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let hyper = NigHyper::default();
    let gauss = Gaussian::new(0.0, 1.0);
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper);

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
    State::from_prior(ftrs, 1.0, &mut rng)
}

#[test]
fn smoke() {
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(10, 2, &mut rng);

    assert_eq!(state.nrows(), 10);
    assert_eq!(state.ncols(), 2);

    state.update(100, None, None, &mut rng);
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

// #[test]
// fn serialize() {
//     let mut rng = rand::thread_rng();
//     let mut state = gen_all_gauss_state(4, 3, &mut rng);

//     let yaml = serde_yaml::to_string(&state).unwrap();
//     println!("{}", yaml);
// }

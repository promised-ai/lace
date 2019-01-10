#[macro_use]
extern crate approx;
extern crate braid;
extern crate rand;
extern crate rv;
extern crate serde_yaml;

use self::rand::Rng;

use self::rv::dist::{Gamma, Gaussian};
use self::rv::traits::Rv;
use braid::cc::Codebook;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::DataContainer;
use braid::cc::DataStore;
use braid::cc::State;
use braid::dist::prior::ng::NigHyper;
use braid::dist::prior::Ng;
use braid::interface::utils::load_states;
use braid::interface::Given;
use braid::Oracle;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let hyper = NigHyper::default();
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

fn get_oracle_from_yaml() -> Oracle {
    let filenames = vec![
        "resources/test/small-state-1.yaml",
        "resources/test/small-state-2.yaml",
        "resources/test/small-state-3.yaml",
    ];
    let states = load_states(filenames);
    let data = DataStore::new(states[0].clone_data());
    Oracle {
        states,
        codebook: Codebook::default(),
        data,
    }
}

fn gen_oracle(nstates: usize) -> Oracle {
    let nrows = 20;
    let ncols = 10;
    let mut rng = rand::thread_rng();
    let states: Vec<State> = (0..nstates)
        .map(|_| gen_all_gauss_state(nrows, ncols, &mut rng))
        .collect();

    let data = DataStore::new(states[0].clone_data());
    Oracle {
        states,
        codebook: Codebook::default(),
        data,
    }
}

#[test]
fn init_from_raw_struct_smoke() {
    let _oracle = gen_oracle(4);
}

#[test]
fn init_from_yaml_files_smoke() {
    let _oracle = get_oracle_from_yaml();
}

#[test]
fn dependence_probability() {
    let oracle = get_oracle_from_yaml();

    assert_relative_eq!(oracle.depprob(0, 1), 1.0 / 3.0, epsilon = 10E-6);
    assert_relative_eq!(oracle.depprob(1, 2), 2.0 / 3.0, epsilon = 10E-6);
    assert_relative_eq!(oracle.depprob(0, 2), 2.0 / 3.0, epsilon = 10E-6);
}

#[test]
fn row_similarity() {
    let oracle = get_oracle_from_yaml();

    let rowsim_01 = (0.5 + 0.5 + 0.0) / 3.0;
    let rowsim_12 = (0.5 + 0.5 + 1.0) / 3.0;
    let rowsim_23 = (1.0 + 0.5 + 1.0) / 3.0;

    assert_relative_eq!(oracle.rowsim(0, 1, None), rowsim_01, epsilon = 10E-6);
    assert_relative_eq!(oracle.rowsim(1, 2, None), rowsim_12, epsilon = 10E-6);
    assert_relative_eq!(oracle.rowsim(2, 3, None), rowsim_23, epsilon = 10E-6);
}

#[test]
fn row_similarity_with_respect_to() {
    let oracle = get_oracle_from_yaml();

    let rowsim_01 = (1.0 + 0.0 + 0.0) / 3.0;
    let rowsim_12 = (0.0 + 1.0 + 1.0) / 3.0;
    let rowsim_23 = (1.0 + 0.0 + 1.0) / 3.0;

    let wrt_cols = vec![0];
    let wrt = Some(&wrt_cols);

    assert_relative_eq!(oracle.rowsim(0, 1, wrt), rowsim_01, epsilon = 10E-6);
    assert_relative_eq!(oracle.rowsim(1, 2, wrt), rowsim_12, epsilon = 10E-6);
    assert_relative_eq!(oracle.rowsim(2, 3, wrt), rowsim_23, epsilon = 10E-6);
}

// Simulation tests
// ================
#[test]
fn simulate_single_col_without_given_size_check() {
    let oracle = get_oracle_from_yaml();
    let mut rng = rand::thread_rng();

    let xs = oracle.simulate(&vec![0], &Given::Nothing, 14, None, &mut rng);

    assert_eq!(xs.len(), 14);
    assert!(xs.iter().all(|x| x.len() == 1));
}

#[test]
fn simulate_multi_col_without_given_size_check() {
    let oracle = get_oracle_from_yaml();
    let mut rng = rand::thread_rng();

    let xs = oracle.simulate(&vec![0, 1], &Given::Nothing, 14, None, &mut rng);

    assert_eq!(xs.len(), 14);
    assert!(xs.iter().all(|x| x.len() == 2));
}

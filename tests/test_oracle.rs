#[macro_use] extern crate approx;
extern crate rand;
extern crate serde_yaml;
extern crate braid;


use self::rand::Rng;

use braid::cc::DataContainer;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::State;
use braid::Oracle;
use braid::dist::Gaussian;
use braid::dist::traits::RandomVariate;
use braid::dist::prior::NormalInverseGamma;



fn gen_col(id: usize, n: usize, mut rng: &mut Rng) -> ColModel {
    let gauss = Gaussian::new(0.0, 1.0);
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);

    let ftr = Column::new(id, data, prior);
    ColModel::Continuous(ftr)
}


fn gen_all_gauss_state(nrows: usize, ncols: usize, mut rng: &mut Rng) -> State {
    let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
    for i in 0..ncols {
        ftrs.push(gen_col(i, nrows, &mut rng));
    }
    State::from_prior(ftrs, 1.0, &mut rng)
}


fn get_oracle_from_yaml() -> Oracle {
    let filenames = vec![
        "resources/test/small-state-1.yaml",
        "resources/test/small-state-2.yaml",
        "resources/test/small-state-3.yaml"];

    Oracle::from_yaml(filenames)
}


fn gen_oracle(nstates: usize) -> Oracle {
    let nrows = 20;
    let ncols = 10;
    let mut rng = rand::thread_rng();
    let states: Vec<State> = (0..nstates)
        .map(|_| gen_all_gauss_state(nrows, ncols, &mut rng))
        .collect();

    Oracle{states: states}
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

    assert_relative_eq!(oracle.depprob(0, 1).unwrap(), 1.0/3.0, epsilon=10E-6);
    assert_relative_eq!(oracle.depprob(1, 2).unwrap(), 2.0/3.0, epsilon=10E-6);
    assert_relative_eq!(oracle.depprob(0, 2).unwrap(), 2.0/3.0, epsilon=10E-6);
}


#[test]
fn row_similarity() {
    let oracle = get_oracle_from_yaml();

    let rowsim_01 = (0.5 + 0.5 + 0.0)/3.0;
    let rowsim_12 = (0.5 + 0.5 + 1.0)/3.0;
    let rowsim_23 = (1.0 + 0.5 + 1.0)/3.0;

    assert_relative_eq!(oracle.rowsim(0, 1, None).unwrap(),
                        rowsim_01, epsilon=10E-6);
    assert_relative_eq!(oracle.rowsim(1, 2, None).unwrap(),
                        rowsim_12, epsilon=10E-6);
    assert_relative_eq!(oracle.rowsim(2, 3, None).unwrap(),
                        rowsim_23, epsilon=10E-6);
}


#[test]
fn row_similarity_with_respect_to() {
    let oracle = get_oracle_from_yaml();

    let rowsim_01 = (1.0 + 0.0 + 0.0)/3.0;
    let rowsim_12 = (0.0 + 1.0 + 1.0)/3.0;
    let rowsim_23 = (1.0 + 0.0 + 1.0)/3.0;


    let wrt_cols = vec![0];
    let wrt = Some(&wrt_cols);

    assert_relative_eq!(oracle.rowsim(0, 1, wrt).unwrap(),
                        rowsim_01, epsilon=10E-6);
    assert_relative_eq!(oracle.rowsim(1, 2, wrt).unwrap(),
                        rowsim_12, epsilon=10E-6);
    assert_relative_eq!(oracle.rowsim(2, 3, wrt).unwrap(),
                        rowsim_23, epsilon=10E-6);
}


// Simulation tests
// ================
#[test]
fn simulate_single_col_without_given_size_check() {
    let oracle = get_oracle_from_yaml();
    let mut rng = rand::thread_rng();

    let xs = oracle.simulate(&vec![0], &None, 14, &mut rng).unwrap();

    assert_eq!(xs.len(), 14);
    assert!(xs.iter().all(|x| x.len() == 1));
}


#[test]
fn simulate_multi_col_without_given_size_check() {
    let oracle = get_oracle_from_yaml();
    let mut rng = rand::thread_rng();

    let xs = oracle.simulate(&vec![0, 1], &None, 14, &mut rng).unwrap();

    assert_eq!(xs.len(), 14);
    assert!(xs.iter().all(|x| x.len() == 2));
}

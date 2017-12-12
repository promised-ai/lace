#[macro_use] extern crate approx;
extern crate rand;
extern crate serde_yaml;
extern crate braid;


use std::fs::File;
use std::path::Path;
use std::io::Read;

use self::rand::Rng;

use braid::cc::DataContainer;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::State;
use braid::cc::Teller;
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


fn get_teller_from_yaml() -> Teller {
    // 3 States
    let paths = vec![
        Path::new("resources/test/small-state-1.yaml"),
        Path::new("resources/test/small-state-2.yaml"),
        Path::new("resources/test/small-state-3.yaml")];

    let mut states: Vec<State> = Vec::with_capacity(3);

    paths.iter().for_each(|path| {
        let mut file = File::open(&path).unwrap();
        let mut yaml = String::new();
        file.read_to_string(&mut yaml);
        states.push(serde_yaml::from_str(&yaml).unwrap());
    });

    Teller{states: states, nrows: 4, ncols: 3, nstates: 3}
}


fn gen_teller(nstates: usize) -> Teller {
    let nrows = 20;
    let ncols = 10;
    let mut rng = rand::thread_rng();
    let states: Vec<State> = (0..nstates)
        .map(|_| gen_all_gauss_state(20, 10, &mut rng))
        .collect();

    Teller{states: states, nrows: nrows, ncols: ncols, nstates: nstates}
}


#[test]
fn init_from_raw_struct_smoke() {
    let teller = gen_teller(4);
}


#[test]
fn init_from_yaml_files_smoke() {
    let teller = get_teller_from_yaml();
}


#[test]
fn dependence_probability() {
    let teller = get_teller_from_yaml();

    assert_relative_eq!(teller.depprob(0, 1), 1.0/3.0, epsilon=10E-6);
    assert_relative_eq!(teller.depprob(1, 2), 2.0/3.0, epsilon=10E-6);
    assert_relative_eq!(teller.depprob(0, 2), 2.0/3.0, epsilon=10E-6);
}


#[test]
fn row_similarity() {
    let teller = get_teller_from_yaml();

    let rowsim_01 = (0.5 + 0.5 + 0.0)/3.0;
    let rowsim_12 = (0.5 + 0.5 + 1.0)/3.0;
    let rowsim_23 = (1.0 + 0.5 + 1.0)/3.0;

    assert_relative_eq!(teller.rowsim(0, 1, None), rowsim_01, epsilon=10E-6);
    assert_relative_eq!(teller.rowsim(1, 2, None), rowsim_12, epsilon=10E-6);
    assert_relative_eq!(teller.rowsim(2, 3, None), rowsim_23, epsilon=10E-6);
}


#[test]
fn row_similarity_with_respect_to() {
    let teller = get_teller_from_yaml();

    let rowsim_01 = (1.0 + 0.0 + 0.0)/3.0;
    let rowsim_12 = (0.0 + 1.0 + 1.0)/3.0;
    let rowsim_23 = (1.0 + 0.0 + 1.0)/3.0;


    let wrt_cols = vec![0];
    let wrt = Some(&wrt_cols);

    assert_relative_eq!(teller.rowsim(0, 1, wrt), rowsim_01, epsilon=10E-6);
    assert_relative_eq!(teller.rowsim(1, 2, wrt), rowsim_12, epsilon=10E-6);
    assert_relative_eq!(teller.rowsim(2, 3, wrt), rowsim_23, epsilon=10E-6);
}

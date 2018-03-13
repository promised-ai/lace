extern crate braid;
extern crate rand;
extern crate serde_yaml;

use self::rand::Rng;
use braid::cc::DataContainer;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::State;
use braid::dist::Gaussian;
use braid::dist::traits::RandomVariate;
use braid::dist::prior::NormalInverseGamma;
use braid::dist::prior::nig::NigHyper;

fn gen_col(id: usize, n: usize, mut rng: &mut Rng) -> ColModel {
    let hyper = NigHyper::default();
    let gauss = Gaussian::new(0.0, 1.0);
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper);

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

#[test]
fn smoke() {
    let mut rng = rand::thread_rng();
    let mut state = gen_all_gauss_state(10, 2, &mut rng);

    assert_eq!(state.nrows(), 10);
    assert_eq!(state.ncols(), 2);

    state.update(100, &mut rng);
}

// #[test]
// fn serialize() {
//     let mut rng = rand::thread_rng();
//     let mut state = gen_all_gauss_state(4, 3, &mut rng);

//     let yaml = serde_yaml::to_string(&state).unwrap();
//     println!("{}", yaml);
// }

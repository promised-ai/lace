
extern crate rand;
extern crate braid;


use self::rand::Rng;
use braid::cc::DataContainer;
use braid::cc::ColModel;
use braid::cc::Column;
use braid::cc::State;
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

fn gen_all_gauss_state(nrows: usize, ncols: usize) -> State<rand::ThreadRng> {
    let mut rng = rand::thread_rng();
    let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
    for i in 0..ncols {
        ftrs.push(gen_col(i, nrows, &mut rng));
    }
    State::from_prior(ftrs, 1.0, rng)
}

#[test]
fn smoke() {
    let mut state = gen_all_gauss_state(10, 2);
    state.update(100);
}

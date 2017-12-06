
extern crate rand;
extern crate braid;


use self::rand::Rng;
use braid::cc::DataContainer;
use braid::cc::Feature;
use braid::cc::Column;
use braid::cc::View;
use braid::cc::State;
use braid::cc::view::RowAssignAlg;
use braid::dist::Gaussian;
use braid::dist::traits::RandomVariate;
use braid::dist::prior::NormalInverseGamma;


type GaussCol = Column<f64, Gaussian, NormalInverseGamma>;


fn gen_col(id: usize, n: usize, mut rng: &mut Rng) -> GaussCol {
    let gauss = Gaussian::new(0.0, 1.0);
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);

    Column::new(id, data, prior)
}

fn gen_all_gauss_state(nrows: usize, ncols: usize) -> State<rand::ThreadRng> {
    let mut rng = rand::thread_rng();
    let mut ftrs: Vec<Box<Feature>> = Vec::with_capacity(ncols);
    for i in 0..ncols {
        ftrs.push(Box::new(gen_col(i, nrows, &mut rng)));
    }
    State::from_prior(ftrs, 1.0, rng)
}

#[test]
fn smoke() {
    let mut state = gen_all_gauss_state(10, 2);
    state.update(100);
}

#![feature(test)]

extern crate braid;
extern crate rand;
extern crate test;

use rand::{Rng, XorShiftRng};
use test::Bencher;

use braid::cc::{Column, Feature, View, DataContainer};
use braid::dist::Gaussian;
use braid::cc::view::RowAssignAlg;
use braid::dist::traits::RandomVariate;
use braid::dist::prior::NormalInverseGamma;


type GaussCol = Column<f64, Gaussian, NormalInverseGamma>;


fn gendata_gauss(id: usize, n: usize, mut rng: &mut Rng) -> GaussCol {
    let mut xs = Gaussian::new(-3.0, 1.0).sample(n, &mut rng);
    let mut ys = Gaussian::new(3.0, 1.0).sample(n, &mut rng);

    xs.append(&mut ys); 

    let data = DataContainer::new(xs);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);

    Column::new(id, data, prior)
}


#[bench]
fn run_100row_01col_benchmark(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();

    let mut ftrs: Vec<Box<Feature>> = vec![];
    ftrs.insert(0, Box::new(gendata_gauss(0, 100, &mut rng)));

    let mut view = View::new(ftrs, 1.0, &mut rng);

    b.iter(|| {
        test::black_box(view.update(1, RowAssignAlg::FiniteCpu, &mut rng));
    });
}

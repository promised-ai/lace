#![feature(test)]
extern crate test;
extern crate rand;
extern crate braid;

use test::Bencher;
use rand::XorShiftRng;
use braid::cc::container::DataContainer;
use braid::dist::Gaussian;
use braid::dist::traits::{RandomVariate, AccumScore};


#[bench]
fn gauss_accum_score_serial(b: &mut Bencher) {
    let n = 100_000;

    let mut rng = XorShiftRng::new_unseeded();

    let xs = Gaussian::new(-3.0, 1.0).sample(n, &mut rng);
    let data = DataContainer::new(xs);
    let gauss = Gaussian::new(0.0, 1.0);

    let mut scores: Vec<f64> = vec![0.0; n];

    b.iter(|| {
        gauss.accum_score(&mut scores, &data.data, &data.present);
    });
}


#[bench]
fn gauss_accum_score_parallel(b: &mut Bencher) {
    let n = 100_000;

    let mut rng = XorShiftRng::new_unseeded();

    let xs = Gaussian::new(-3.0, 1.0).sample(n, &mut rng);
    let data = DataContainer::new(xs);
    let gauss = Gaussian::new(0.0, 1.0);

    let mut scores: Vec<f64> = vec![0.0; n];

    b.iter(|| {
        gauss.accum_score_par(&mut scores, &data.data, &data.present);
    });
}

#![feature(test)]

extern crate criterion;
extern crate braid;
extern crate rand;
extern crate test;

use criterion::Criterion;
use test::Bencher;
use braid::misc;
use rand::XorShiftRng;


#[bench]
fn pflip(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();
    let weights: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    b.iter(|| {
        test::black_box(misc::pflip(&weights, &mut rng));
    });
}


#[bench]
fn log_pflip(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();
    let weights: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    b.iter(|| {
        test::black_box(misc::log_pflip(&weights, &mut rng));
    });
}


#[test]
fn massflip() {
    fn routine(b: &mut criterion::Bencher) {
        let mut rng = XorShiftRng::new_unseeded();
        b.iter_with_setup(|| {let xs: Vec<Vec<f64>> = vec![vec![0.0; 5]; 2500]; xs},
                          |w| {misc::massflip(w, &mut rng);});
    }
    Criterion::default().bench_function("massflip", routine);
}


#[test]
fn massflip_long_parallel() {
    fn routine(b: &mut criterion::Bencher) {
        let mut rng = XorShiftRng::new_unseeded();
        b.iter_with_setup(|| {let xs: Vec<Vec<f64>> = vec![vec![0.0; 5]; 2500]; xs},
                          |w| {misc::massflip_par(w, &mut rng);});
    }
    Criterion::default().bench_function("massflip_par", routine);
}


#[bench]
fn massflip_long_serial(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();
    b.iter(|| {
        let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25000];
        test::black_box(misc::massflip(log_weights, &mut rng));
    });
}


#[bench]
fn transpose(b: &mut Bencher) {
    let x: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25000];
    b.iter(|| {
        test::black_box(misc::transpose(&x));
    });
}

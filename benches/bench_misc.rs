#![feature(test)]

extern crate braid;
extern crate rand;
extern crate test;

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


#[bench]
fn massflip(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();
    b.iter(|| {
        let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25];
        test::black_box(misc::massflip(log_weights, &mut rng));
    });
}


#[bench]
fn massflip_long_parallel(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();
    b.iter(|| {
        let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25000];
        test::black_box(misc::massflip_par(log_weights, &mut rng));
    });
}


#[bench]
fn massflip_long_serial(b: &mut Bencher) {
    let mut rng = XorShiftRng::new_unseeded();
    b.iter(|| {
        let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25000];
        test::black_box(misc::massflip(log_weights, &mut rng));
    });
}

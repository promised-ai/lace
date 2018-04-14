#![feature(test)]

#[macro_use]
extern crate criterion;

extern crate braid;
extern crate rand;
extern crate test;

use braid::misc;
use criterion::Criterion;
use rand::XorShiftRng;

fn pflip(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        b.iter_with_setup(
            || {
                let rng = XorShiftRng::new_unseeded();
                let weights: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
                (weights, rng)
            },
            |mut fxtr| {
                misc::pflip(&fxtr.0, 1, &mut fxtr.1);
            },
        );
    };
    c.bench_function("pflip", routine);
}

fn log_pflip(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        b.iter_with_setup(
            || {
                let rng = XorShiftRng::new_unseeded();
                let weights: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
                (weights, rng)
            },
            |mut fxtr| {
                misc::log_pflip(&fxtr.0, &mut fxtr.1);
            },
        );
    };
    c.bench_function("log pflip", routine);
}

fn massflip(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        let mut rng = XorShiftRng::new_unseeded();
        b.iter_with_setup(
            || {
                let xs: Vec<Vec<f64>> = vec![vec![0.0; 5]; 2500];
                xs
            },
            |w| {
                misc::massflip(w, &mut rng);
            },
        );
    };
    c.bench_function("massflip", routine);
}

fn massflip_long_parallel(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        let mut rng = XorShiftRng::new_unseeded();
        b.iter_with_setup(
            || {
                let xs: Vec<Vec<f64>> = vec![vec![0.0; 5]; 2500];
                xs
            },
            |w| {
                misc::massflip_par(w, &mut rng);
            },
        );
    };
    c.bench_function("massflip log parallel", routine);
}

fn massflip_long_serial(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        let mut rng = XorShiftRng::new_unseeded();
        b.iter_with_setup(
            || {
                let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25000];
                log_weights
            },
            |w| {
                test::black_box(misc::massflip(w, &mut rng));
            },
        );
    };
    c.bench_function("massflip log serial", routine);
}

fn transpose(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        b.iter_with_setup(
            || {
                let m: Vec<Vec<f64>> = vec![vec![0.0; 5]; 25000];
                m
            },
            |m| {
                test::black_box(misc::transpose(&m));
            },
        );
    }
    c.bench_function("transpose", routine);
}

criterion_group!(
    benches,
    pflip,
    log_pflip,
    massflip,
    massflip_long_parallel,
    massflip_long_serial,
    transpose
);
criterion_main!(benches);

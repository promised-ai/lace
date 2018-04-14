#[macro_use]
extern crate criterion;
extern crate braid;
extern crate rand;

use braid::cc::container::DataContainer;
use braid::dist::Gaussian;
use braid::dist::traits::{AccumScore, RandomVariate};
use criterion::Criterion;
use rand::XorShiftRng;

fn gauss_accum_score_serial(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        b.iter_with_setup(
            || {
                let n = 100_000;
                let mut rng = XorShiftRng::new_unseeded();
                let xs = Gaussian::new(-3.0, 1.0).sample(n, &mut rng);
                let data = DataContainer::new(xs);
                let gauss = Gaussian::new(0.0, 1.0);
                let mut scores: Vec<f64> = vec![0.0; n];
                (data, scores, gauss)
            },
            |mut f| {
                f.2
                    .accum_score(&mut f.1, &f.0.data, &f.0.present);
            },
        );
    };
    c.bench_function("gauss acculate scores serial", routine);
}

fn gauss_accum_score_parallel(c: &mut Criterion) {
    fn routine(b: &mut criterion::Bencher) {
        b.iter_with_setup(
            || {
                let n = 100_000;
                let mut rng = XorShiftRng::new_unseeded();
                let xs = Gaussian::new(-3.0, 1.0).sample(n, &mut rng);
                let data = DataContainer::new(xs);
                let gauss = Gaussian::new(0.0, 1.0);
                let mut scores: Vec<f64> = vec![0.0; n];
                (data, scores, gauss)
            },
            |mut f| {
                f.2
                    .accum_score_par(&mut f.1, &f.0.data, &f.0.present);
            },
        );
    };
    c.bench_function("gauss acculate scores parallel", routine);
}

criterion_group!(
    benches,
    gauss_accum_score_serial,
    gauss_accum_score_parallel
);
criterion_main!(benches);

use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::ParameterizedBenchmark;
use criterion::{criterion_group, criterion_main};

use braid_flippers::*;
use rand::{FromEntropy, Rng};
use rand_xoshiro::Xoshiro256Plus;

fn gen_log_weights(n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    vec![vec![0.5; n_cols]; n_rows]
}

fn bench_compare_5_rows(c: &mut Criterion) {
    c.bench(
        "Compare Parallel vs Serial Massflip (10 cols)",
        ParameterizedBenchmark::new(
            "serial",
            |b, &n_rows| {
                let mut rng = Xoshiro256Plus::from_entropy();
                b.iter_batched(
                    || gen_log_weights(n_rows, 10),
                    |logps| {
                        let _ixs = black_box(massflip_ser(logps, &mut rng));
                    },
                    BatchSize::LargeInput,
                )
            },
            vec![100, 500, 1000, 5000, 10_000],
        )
        .with_function("for_each", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights(n_rows, 10),
                |logps| {
                    let _ixs = black_box(massflip_ser_fe(logps, &mut rng));
                },
                BatchSize::LargeInput,
            )
        })
        .with_function("paralllel", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights(n_rows, 10),
                |logps| {
                    let _ixs = black_box(massflip_par(logps, &mut rng));
                },
                BatchSize::LargeInput,
            )
        }),
    );
}

criterion_group!(benches, bench_compare_5_rows);
criterion_main!(benches);

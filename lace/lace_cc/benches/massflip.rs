use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::ParameterizedBenchmark;
use criterion::{criterion_group, criterion_main};

use lace_flippers::*;
use lace_utils::Matrix;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn gen_log_weights_mat(n_rows: usize, n_cols: usize) -> Matrix<f64> {
    let vecs = vec![vec![0.5; n_rows]; n_cols];
    Matrix::from_vecs(vecs)
}

fn bench_compare_5_rows(c: &mut Criterion) {
    c.bench(
        "Compare Parallel vs Serial Massflip (10 cols)",
        ParameterizedBenchmark::new(
            "serial",
            |b, &n_rows| {
                let mut rng = Xoshiro256Plus::from_entropy();
                b.iter_batched(
                    || gen_log_weights_mat(n_rows, 10),
                    |logps| {
                        // massflip_mat and massflip_mat_par transpose inside
                        let ixs = black_box(massflip_mat(
                            &logps.implicit_transpose(),
                            &mut rng,
                        ));
                        assert_eq!(ixs.len(), n_rows);
                    },
                    BatchSize::LargeInput,
                )
            },
            vec![100, 500, 1000, 5000, 10_000, 50_000],
        )
        .with_function("parallel", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |logps| {
                    let ixs = black_box(massflip_mat_par(
                        &logps.implicit_transpose(),
                        &mut rng,
                    ));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        }),
    );
}

fn bench_compare_5_rows_slice(c: &mut Criterion) {
    c.bench(
        "Compare Parallel vs Serial Massflip Slice (10 cols)",
        ParameterizedBenchmark::new(
            "serial",
            |b, &n_rows| {
                let mut rng = Xoshiro256Plus::from_entropy();
                b.iter_batched(
                    || gen_log_weights_mat(n_rows, 10),
                    |logps| {
                        let ixs = black_box(massflip_slice_mat(
                            &logps.implicit_transpose(),
                            &mut rng,
                        ));
                        assert_eq!(ixs.len(), n_rows);
                    },
                    BatchSize::LargeInput,
                )
            },
            vec![100, 500, 1000, 5000, 10_000, 50_000],
        )
        .with_function("parallel par", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |logps| {
                    let ixs = black_box(massflip_slice_mat_par(
                        &logps.implicit_transpose(),
                        &mut rng,
                    ));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        }),
    );
}

criterion_group!(benches, bench_compare_5_rows, bench_compare_5_rows_slice);
criterion_main!(benches);

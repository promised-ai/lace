use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::ParameterizedBenchmark;
use criterion::{criterion_group, criterion_main};

use braid_flippers::*;
use braid_utils::Matrix;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

pub fn transpose(mat_in: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let nrows = mat_in.len();
    let ncols = mat_in[0].len();
    let mut mat_out: Vec<Vec<f64>> = vec![vec![0.0; nrows]; ncols];

    for (i, row) in mat_in.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            mat_out[j][i] = x;
        }
    }

    mat_out
}

fn gen_log_weights(n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    vec![vec![0.5; n_rows]; n_cols]
}

fn gen_log_weights_mat(n_rows: usize, n_cols: usize) -> Matrix<f64> {
    let vecs = vec![vec![0.5; n_rows]; n_cols];
    Matrix::from_vecs(&vecs)
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
                        let logps_t = transpose(&logps);
                        let _ixs = black_box(massflip_ser(logps_t, &mut rng));
                    },
                    BatchSize::LargeInput,
                )
            },
            vec![100, 500, 1000, 5000, 10_000, 50_000],
        )
        .with_function("for_each", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights(n_rows, 10),
                |logps| {
                    let logps_t = transpose(&logps);
                    let _ixs = black_box(massflip_ser_fe(logps_t, &mut rng));
                },
                BatchSize::LargeInput,
            )
        })
        .with_function("matrix par", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |mut logps| {
                    logps.transpose();
                    let ixs = black_box(massflip_mat_par(&logps, &mut rng));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        })
        .with_function("matrix", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |mut logps| {
                    // massflip_mat and massflip_mat_par transpose inside
                    logps.transpose();
                    let ixs = black_box(massflip_mat(&logps, &mut rng));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        })
        .with_function("paralllel", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights(n_rows, 10),
                |logps| {
                    let logps_t = transpose(&logps);
                    let _ixs = black_box(massflip_par(logps_t, &mut rng));
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
            "normal",
            |b, &n_rows| {
                let mut rng = Xoshiro256Plus::from_entropy();
                b.iter_batched(
                    || gen_log_weights(n_rows, 10),
                    |logps| {
                        let logps_t = transpose(&logps);
                        let _ixs = black_box(massflip_slice(logps_t, &mut rng));
                    },
                    BatchSize::LargeInput,
                )
            },
            vec![100, 500, 1000, 5000, 10_000, 50_000],
        )
        .with_function("matrix", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |mut logps| {
                    logps.transpose();
                    let ixs = black_box(massflip_slice_mat(&logps, &mut rng));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        })
        .with_function("matrix par", |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |mut logps| {
                    logps.transpose();
                    let ixs =
                        black_box(massflip_slice_mat_par(&logps, &mut rng));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        }),
    );
}

criterion_group!(benches, bench_compare_5_rows, bench_compare_5_rows_slice);
criterion_main!(benches);

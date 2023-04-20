use criterion::black_box;
use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use lace_cc::massflip::massflip_mat;
use lace_cc::massflip::massflip_mat_par;
use lace_cc::massflip::massflip_slice_mat;
use lace_cc::massflip::massflip_slice_mat_par;
use lace_utils::Matrix;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn gen_log_weights_mat(n_rows: usize, n_cols: usize) -> Matrix<f64> {
    let vecs = vec![vec![0.5; n_rows]; n_cols];
    Matrix::from_vecs(vecs)
}

fn bench_compare_5_rows(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("Compare Parallel vs Serial Massflip (10 cols)");

    let parameters: Vec<usize> = vec![100, 500, 1000, 5000, 10_000, 50_000];

    for n_rows in parameters {
        let serial_id = BenchmarkId::new("serial", n_rows);
        group.bench_with_input(serial_id, &n_rows, |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |logps| {
                    // massflip_mat and massflip_mat_par transpose inside
                    let ixs = black_box(massflip_mat(
                        logps.implicit_transpose(),
                        &mut rng,
                    ));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        });

        let parallel_id = BenchmarkId::new("parallel", n_rows);
        group.bench_with_input(parallel_id, &n_rows, |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |logps| {
                    let ixs = black_box(massflip_mat_par(
                        logps.implicit_transpose(),
                        &mut rng,
                    ));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        });
    }
}

fn bench_compare_5_rows_slice(c: &mut Criterion) {
    let mut group = c
        .benchmark_group("Compare Parallel vs Serial Massflip Slice (10 cols)");

    let parameters: Vec<usize> = vec![100, 500, 1000, 5000, 10_000, 50_000];

    for n_rows in parameters {
        let serial_id = BenchmarkId::new("serial", n_rows);
        group.bench_with_input(serial_id, &n_rows, |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |logps| {
                    let ixs = black_box(massflip_slice_mat(
                        logps.implicit_transpose(),
                        &mut rng,
                    ));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        });

        let parallel_id = BenchmarkId::new("parallel par", n_rows);
        group.bench_with_input(parallel_id, &n_rows, |b, &n_rows| {
            let mut rng = Xoshiro256Plus::from_entropy();
            b.iter_batched(
                || gen_log_weights_mat(n_rows, 10),
                |logps| {
                    let ixs = black_box(massflip_slice_mat_par(
                        logps.implicit_transpose(),
                        &mut rng,
                    ));
                    assert_eq!(ixs.len(), n_rows);
                },
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(benches, bench_compare_5_rows, bench_compare_5_rows_slice);
criterion_main!(benches);

use lace_stats::seq::*;

use criterion::{black_box, BenchmarkId};
use criterion::{
    criterion_group, criterion_main, Criterion, 
};

fn bench_halton(c: &mut Criterion) {
    c.bench_function("Halton(2) 1k numbers", |b| {
        b.iter(|| {
            let seq = HaltonSeq::new(2);
            let xs: Vec<f64> = seq.take(1_000).collect();
            black_box(xs)
        })
    });
}

fn bench_sobol(c: &mut Criterion) {
    c.bench_function("Sobol(2) 1k numbers", |b| {
        b.iter(|| {
            let seq = SobolSeq::new(1);
            let xs: Vec<Vec<f64>> = seq.take(1_000).collect();
            black_box(xs);
        })
    });
}

fn seq_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("Seq compare");

    let parameters: Vec<usize> = vec![10, 1_000, 100_000];

    for chunk_size in parameters {
        let halton_id = BenchmarkId::new("Halton", chunk_size);
        group.bench_with_input(halton_id, &chunk_size, 
            |b, i| {
                b.iter(|| {
                    let seq = HaltonSeq::new(2);
                    let xs: Vec<f64> = seq.take(*i).collect();
                    black_box(xs)
                })
            },
        );

        let sobol_id = BenchmarkId::new("Sobol", chunk_size);
        group.bench_with_input(sobol_id, &chunk_size, 
            |b, i| {
                b.iter(|| {
                    let seq = SobolSeq::new(1);
                    let xs: Vec<Vec<f64>> = seq.take(*i).collect();
                    black_box(xs);
                })
            }
        );
    }
}

criterion_group!(seq_benches, bench_halton, bench_sobol, seq_compare,);
criterion_main!(seq_benches);

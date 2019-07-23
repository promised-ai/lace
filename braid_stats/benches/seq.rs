use braid_stats::seq::*;

use criterion::black_box;
use criterion::{Criterion, criterion_group, criterion_main, ParameterizedBenchmark};


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
    c.bench("Seq compare",
        ParameterizedBenchmark::new(
            "Halton", |b, i| b.iter(|| {
                let seq = HaltonSeq::new(2);
                let xs: Vec<f64> = seq.take(*i).collect();
                black_box(xs)
            }),
            vec![10, 1_000, 100_000]
        )
        .with_function("Sobol", |b, i| {
            b.iter(|| {
                let seq = SobolSeq::new(1);
                let xs: Vec<Vec<f64>> = seq.take(*i).collect();
                black_box(xs);
            })
        }),
    );
}


criterion_group!(seq_benches, bench_halton, bench_sobol, seq_compare, );
criterion_main!(seq_benches);

use braid_stats::seq::*;

use criterion::black_box;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

fn bench_halton(c: &mut Criterion) {
    c.bench_function("Halton(2) 1k numbers", |b| {
        b.iter(|| {
            let seq = HaltonSeq::new(2);
            let xs: Vec<f64> = seq.take(1_000).collect();
            black_box(xs)
        })
    });
}

criterion_group!(seq_benches, bench_halton,);
criterion_main!(seq_benches);

use criterion::black_box;
use criterion::{
    criterion_group, criterion_main, BatchSize, Criterion,
    ParameterizedBenchmark,
};

use lace_stats::seq::SobolSeq;

fn u2s_alloc(mut uvec: Vec<f64>) -> Vec<f64> {
    let n = uvec.len();
    uvec[0] = 0.0;
    uvec[n - 1] = 1.0;
    uvec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // turn off mutability
    let uvec = uvec;

    let mut z = Vec::with_capacity(n);
    for i in 1..n {
        z.push(uvec[i] - uvec[i - 1]);
    }

    z
}

fn u2s_update(mut uvec: Vec<f64>) -> Vec<f64> {
    let n = uvec.len();
    uvec[n - 1] = 1.0;
    uvec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut um = uvec[0];

    for i in 1..n {
        let diff = uvec[i] - um;
        um = uvec[i];
        uvec[i] = diff;
    }

    uvec
}

fn bench_compare(c: &mut Criterion) {
    c.bench(
        "Sobol to 3-simplex over NDims",
        ParameterizedBenchmark::new(
            "new alloc",
            |b, &dims| {
                let mut sobol = SobolSeq::new(dims + 1);
                b.iter_batched(
                    || sobol.next().unwrap(),
                    |x| black_box(u2s_alloc(x)),
                    BatchSize::SmallInput,
                )
            },
            vec![3_usize, 5_usize, 10_usize, 20_usize, 30_usize],
        )
        .with_function("update inplace", |b, &dims| {
            let mut sobol = SobolSeq::new(dims);
            b.iter_batched(
                || sobol.next().unwrap(),
                |x| black_box(u2s_update(x)),
                BatchSize::SmallInput,
            )
        }),
    );
}

criterion_group!(simplex_benches, bench_compare,);
criterion_main!(simplex_benches);

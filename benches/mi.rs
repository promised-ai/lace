use braid::examples::Example;
use braid::{MiType, OracleT};
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

const MI_N: usize = 10_000;

fn get_col_pairs(ncols: usize) -> Vec<(usize, usize)> {
    let mut col_pairs: Vec<(usize, usize)> = vec![];
    for i in 0..ncols {
        for j in i..ncols {
            col_pairs.push((i, j));
        }
    }
    col_pairs
}

fn bench_manual_mi(c: &mut Criterion) {
    c.bench_function("manual MI animals", |b| {
        let oracle = Example::Animals.oracle().unwrap();
        let col_pairs = get_col_pairs(oracle.ncols());
        b.iter(|| {
            for (col_a, col_b) in col_pairs.iter() {
                let _mi = black_box(oracle.mi(
                    *col_a,
                    *col_b,
                    MI_N,
                    MiType::UnNormed,
                ));
            }
        })
    });
}

fn bench_pw_mi(c: &mut Criterion) {
    c.bench_function("pairwise MI animals", |b| {
        let oracle = Example::Animals.oracle().unwrap();
        let col_pairs = get_col_pairs(oracle.ncols());
        b.iter(|| {
            let _mi =
                black_box(oracle.mi_pw(&col_pairs, MI_N, MiType::UnNormed));
        })
    });
}

criterion_group!(mi_benches, bench_manual_mi, bench_pw_mi);

criterion_main!(mi_benches);

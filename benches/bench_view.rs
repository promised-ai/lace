#[macro_use]
extern crate criterion;

extern crate braid;
extern crate rand;

use criterion::Criterion;
use rand::{Rng, XorShiftRng};

use braid::cc::view::RowAssignAlg;
use braid::cc::{ColModel, Column, DataContainer, View};
use braid::dist::prior::NormalInverseGamma;
use braid::dist::traits::RandomVariate;
use braid::dist::Gaussian;

fn gen_gauss_col(id: usize, n: usize, mut rng: &mut Rng) -> ColModel {
    let xs = Gaussian::new(-3.0, 1.0).sample(n, &mut rng);
    let data = DataContainer::new(xs);
    let prior = NormalInverseGamma::from_data(&data.data, &mut rng);

    ColModel::Continuous(Column::new(id, data, prior))
}

fn gen_gauss_view(nrows: usize, ncols: usize, mut rng: &mut Rng) -> View {
    let ftrs: Vec<ColModel> = (0..ncols)
        .map(|id| gen_gauss_col(id, nrows, &mut rng))
        .collect();
    View::new(ftrs, 1.0, &mut rng)
}

fn bench_gauss_view_reassign_finite_cpu(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "gauss view finite CPU reassign",
        |b, &&n| {
            b.iter_with_setup(
                || {
                    let mut rng = XorShiftRng::new_unseeded();
                    let view = gen_gauss_view(n, 5, &mut rng);
                    (view, rng)
                },
                |mut fxtr| {
                    let alg = RowAssignAlg::FiniteCpu;
                    fxtr.0.reassign(alg, &mut fxtr.1);
                },
            )
        },
        &[10, 100, 1000, 10000],
    );
}

criterion_group!(benches, bench_gauss_view_reassign_finite_cpu);
criterion_main!(benches);

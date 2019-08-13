use criterion::black_box;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use braid_stats::labeler::{Label, Labeler, LabelerPrior, LabelerSuffStat};
use braid_stats::simplex::SimplexPoint;
use maplit::hashmap;
use rv::data::DataOrSuffStat;
use rv::traits::{ConjugatePrior, Rv};

macro_rules! f_bench {
    ($id: expr, $fn_name: ident, $label: expr, $truth: expr) => {
        fn $fn_name(c: &mut Criterion) {
            c.bench_function($id, |b| {
                let p_world = SimplexPoint::new(vec![0.5, 0.5]).unwrap();
                let labeler = Labeler::new(0.5, 0.5, p_world);
                let label = Label::new($label, $truth);
                b.iter(|| black_box(labeler.f(&label)))
            });
        }
    };
}

f_bench!("labeler f(t|t)", bench_labeler_f_tt, 1, Some(1));
f_bench!("labeler f(t|f)", bench_labeler_f_tf, 1, Some(0));
f_bench!("labeler f(f|t)", bench_labeler_f_ft, 0, Some(1));
f_bench!("labeler f(f|f)", bench_labeler_f_ff, 0, Some(0));
f_bench!("labeler f(t)", bench_labeler_f_tu, 1, None);
f_bench!("labeler f(f)", bench_labeler_f_fu, 0, None);

fn bench_ln_m(c: &mut Criterion) {
    c.bench_function("labeler ln m(x)", |b| {
        let pr = LabelerPrior::standard(2);
        let stat = LabelerSuffStat {
            n: 20,
            counter: hashmap! {
                Label::new(1, Some(1)) => 10,
                Label::new(1, Some(0)) => 5,
                Label::new(0, Some(1)) => 1,
                Label::new(0, Some(0)) => 1,
                Label::new(1, None) => 2,
                Label::new(1, None) => 1,
            },
        };
        let xm = DataOrSuffStat::SuffStat(&stat);
        b.iter(|| black_box(pr.ln_m(&xm)))
    });
}

fn bench_posterior_update(c: &mut Criterion) {
    c.bench_function("labeler posterior draw", |b| {
        let pr = LabelerPrior::standard(2);
        let stat = LabelerSuffStat {
            n: 20,
            counter: hashmap! {
                Label::new(1, Some(1)) => 10,
                Label::new(1, Some(0)) => 5,
                Label::new(0, Some(1)) => 1,
                Label::new(0, Some(0)) => 1,
                Label::new(1, None) => 2,
                Label::new(1, None) => 1,
            },
        };
        let xm = DataOrSuffStat::SuffStat(&stat);
        let post = pr.posterior(&xm);
        let mut rng = rand::thread_rng();
        b.iter(|| black_box(post.draw(&mut rng)))
    });
}

criterion_group!(
    labeler_benches,
    bench_labeler_f_tt,
    bench_labeler_f_tf,
    bench_labeler_f_ft,
    bench_labeler_f_ff,
    bench_labeler_f_tu,
    bench_labeler_f_fu,
    bench_ln_m,
    bench_posterior_update,
);
criterion_main!(labeler_benches);

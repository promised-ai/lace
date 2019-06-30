use criterion::black_box;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use braid_stats::labeler::{Label, Labeler, LabelerPrior, LabelerSuffStat};
use rv::data::DataOrSuffStat;
use rv::traits::{ConjugatePrior, Rv};

macro_rules! f_bench {
    ($id: expr, $fn_name: ident, $label: expr, $truth: expr) => {
        fn $fn_name(c: &mut Criterion) {
            c.bench_function($id, |b| {
                let labeler = Labeler::new(0.5, 0.5, 0.5);
                let label = Label::new($label, $truth);
                b.iter(|| black_box(labeler.f(&label)))
            });
        }
    };
}

f_bench!("labeler f(t|t)", bench_labeler_f_tt, true, Some(true));
f_bench!("labeler f(t|f)", bench_labeler_f_tf, true, Some(false));
f_bench!("labeler f(f|t)", bench_labeler_f_ft, false, Some(true));
f_bench!("labeler f(f|f)", bench_labeler_f_ff, false, Some(false));
f_bench!("labeler f(t)", bench_labeler_f_tu, true, None);
f_bench!("labeler f(f)", bench_labeler_f_fu, false, None);

fn bench_ln_m(c: &mut Criterion) {
    c.bench_function("labeler ln m(x)", |b| {
        let pr = LabelerPrior::default();
        let stat = LabelerSuffStat {
            n: 20,
            n_truth_tt: 10,
            n_truth_tf: 5,
            n_truth_ft: 1,
            n_truth_ff: 1,
            n_unk_t: 2,
            n_unk_f: 1,
        };
        let xm = DataOrSuffStat::SuffStat(&stat);
        b.iter(|| black_box(pr.ln_m(&xm)))
    });
}

fn bench_posterior_update(c: &mut Criterion) {
    c.bench_function("labeler posterior draw", |b| {
        let pr = LabelerPrior::default();
        let stat = LabelerSuffStat {
            n: 20,
            n_truth_tt: 10,
            n_truth_tf: 5,
            n_truth_ft: 1,
            n_truth_ff: 1,
            n_unk_t: 2,
            n_unk_f: 1,
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

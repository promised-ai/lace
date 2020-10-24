use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::traits::Rv;

use braid_stats::UpdatePrior;

fn bench_continuous_prior(c: &mut Criterion) {
    use braid_stats::prior::{Ng, NigHyper};
    use rv::dist::Gaussian;

    c.bench_function("update continuous prior", |b| {
        let hyper = NigHyper::default();
        let mut prior = Ng::new(0.0, 1.0, 1.0, 1.0, hyper);
        let mut rng = Xoshiro256Plus::from_entropy();
        let components: Vec<Gaussian> = prior.sample(50, &mut rng);
        let components_ref: Vec<&Gaussian> = components.iter().collect();
        b.iter(|| {
            let out = prior.update_prior(&components_ref, &mut rng);
            black_box(out);
        })
    });
}

fn bench_categorical_prior(c: &mut Criterion) {
    use braid_stats::prior::{Csd, CsdHyper};
    use rv::dist::Categorical;

    c.bench_function("update categorical prior", |b| {
        let mut rng = Xoshiro256Plus::from_entropy();
        let mut prior = Csd::from_hyper(4, CsdHyper::default(), &mut rng);
        let components: Vec<Categorical> = prior.sample(50, &mut rng);
        let components_ref: Vec<&Categorical> = components.iter().collect();
        b.iter(|| {
            let out = <Csd as UpdatePrior<u8, Categorical>>::update_prior(
                &mut prior,
                &components_ref,
                &mut rng,
            );
            // let out = prior.update_prior(&components_ref, &mut rng);
            black_box(out);
        })
    });
}

fn bench_count_prior(c: &mut Criterion) {
    use braid_stats::prior::{Pg, PgHyper};
    use rv::dist::Poisson;

    c.bench_function("update count prior", |b| {
        let mut rng = Xoshiro256Plus::from_entropy();
        let mut prior = Pg::from_hyper(PgHyper::default(), &mut rng);
        let components: Vec<Poisson> = prior.sample(50, &mut rng);
        let components_ref: Vec<&Poisson> = components.iter().collect();
        b.iter(|| {
            let out = prior.update_prior(&components_ref, &mut rng);
            // let out = prior.update_prior(&components_ref, &mut rng);
            black_box(out);
        })
    });
}

criterion_group!(
    feature_prior_update,
    bench_continuous_prior,
    bench_categorical_prior,
    bench_count_prior
);
criterion_main!(feature_prior_update);

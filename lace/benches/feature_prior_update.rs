use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use lace::stats::UpdatePrior;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::traits::Sampleable;

fn bench_continuous_prior(c: &mut Criterion) {
    use lace::stats::prior::nix::NixHyper;
    use rv::dist::Gaussian;
    use rv::dist::NormalInvChiSquared;

    c.bench_function("update continuous prior", |b| {
        let hyper = NixHyper::default();
        let mut prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);
        let mut rng = Xoshiro256Plus::from_os_rng();
        let components: Vec<Gaussian> = prior.sample(50, &mut rng);
        let components_ref: Vec<&Gaussian> = components.iter().collect();
        b.iter(|| {
            let out = prior.update_prior(&components_ref, &hyper, &mut rng);
            black_box(out);
        })
    });
}

fn bench_categorical_prior(c: &mut Criterion) {
    use lace::stats::prior::csd::CsdHyper;
    use rv::dist::Categorical;
    use rv::dist::SymmetricDirichlet;

    c.bench_function("update categorical prior", |b| {
        let mut rng = Xoshiro256Plus::from_os_rng();
        let hyper = CsdHyper::default();
        let mut prior: SymmetricDirichlet = hyper.draw(4, &mut rng);
        let components: Vec<Categorical> = prior.sample(50, &mut rng);
        let components_ref: Vec<&Categorical> = components.iter().collect();
        b.iter(|| {
            let out = <SymmetricDirichlet as UpdatePrior<
                u8,
                Categorical,
                CsdHyper,
            >>::update_prior(
                &mut prior, &components_ref, &hyper, &mut rng
            );
            // let out = prior.update_prior(&components_ref, &mut rng);
            black_box(out);
        })
    });
}

fn bench_count_prior(c: &mut Criterion) {
    use lace::stats::prior::pg::PgHyper;
    use rv::dist::Gamma;
    use rv::dist::Poisson;

    c.bench_function("update count prior", |b| {
        let mut rng = Xoshiro256Plus::from_os_rng();
        let hyper = PgHyper::default();
        let mut prior: Gamma = hyper.draw(&mut rng);
        let components: Vec<Poisson> = prior.sample(50, &mut rng);
        let components_ref: Vec<&Poisson> = components.iter().collect();
        b.iter(|| {
            let out = prior.update_prior(&components_ref, &hyper, &mut rng);
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

use criterion::BatchSize;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use braid::cc::config::StateUpdateConfig;
use braid::cc::StateBuilder;
use braid_codebook::codebook::ColType;

fn bench_labeler_state(c: &mut Criterion) {
    c.bench_function("20-by-2 (1 views, 5 cats) all-labeler state", |b| {
        let labeler_config = ColType::Labeler {
            pr_h: None,
            pr_k: None,
            pr_world: None,
        };
        let builder = StateBuilder::new()
            .with_rows(20)
            .add_column_configs(2, labeler_config)
            .with_views(1)
            .with_cats(5)
            .with_seed(1337);
        let mut rng = Xoshiro256Plus::seed_from_u64(1337);
        b.iter_batched(
            || builder.clone().build().unwrap(),
            |mut state| {
                let config = StateUpdateConfig::default();
                black_box(state.update(config, &mut rng));
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_categorical_state(c: &mut Criterion) {
    c.bench_function(
        "20-by-2 (1 views, 5 cats) all-Categorical(2) state",
        |b| {
            let labeler_config = ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            };
            let builder = StateBuilder::new()
                .with_rows(20)
                .add_column_configs(2, labeler_config)
                .with_views(1)
                .with_cats(5)
                .with_seed(1337);
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            b.iter_batched(
                || builder.clone().build().unwrap(),
                |mut state| {
                    let config = StateUpdateConfig::default();
                    black_box(state.update(config, &mut rng));
                },
                BatchSize::LargeInput,
            )
        },
    );
}

fn bench_gaussian_state(c: &mut Criterion) {
    c.bench_function("20-by-2 (1 views, 5 cats) all-Gaussian state", |b| {
        let labeler_config = ColType::Continuous { hyper: None };
        let builder = StateBuilder::new()
            .with_rows(20)
            .add_column_configs(2, labeler_config)
            .with_views(1)
            .with_cats(5)
            .with_seed(1337);
        let mut rng = Xoshiro256Plus::seed_from_u64(1337);
        b.iter_batched(
            || builder.clone().build().unwrap(),
            |mut state| {
                let config = StateUpdateConfig::default();
                black_box(state.update(config, &mut rng));
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    state_type_benches,
    bench_labeler_state,
    bench_categorical_state,
    bench_gaussian_state
);
criterion_main!(state_type_benches);

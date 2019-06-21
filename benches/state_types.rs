use criterion::BatchSize;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use braid::cc::config::StateUpdateConfig;
use braid::cc::StateBuilder;
use braid_codebook::codebook::ColType;

macro_rules! state_type_bench {
    ($id: expr, $fn: ident, $config: expr) => {
        fn $fn(c: &mut Criterion) {
            c.bench_function($id, |b| {
                let builder = StateBuilder::new()
                    .with_rows(20)
                    .add_column_configs(2, $config)
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
    }
}

state_type_bench!(
    "20-by-2 (1 views, 5 cats) all-labeler state",
    bench_labeler_state,
    ColType::Labeler {
        pr_h: None,
        pr_k: None,
        pr_world: None,
    }
);

state_type_bench!(
    "20-by-2 (1 views, 5 cats) all-categorical(2) state",
    bench_categorical_state,
    ColType::Categorical {
        k: 2,
        hyper: None,
        value_map: None,
    }
);

state_type_bench!(
    "20-by-2 (1 views, 5 cats) all-gaussian state",
    bench_gaussian_state,
    ColType::Continuous { hyper: None }
);

criterion_group!(
    state_type_benches,
    bench_labeler_state,
    bench_categorical_state,
    bench_gaussian_state
);
criterion_main!(state_type_benches);

use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BatchSize;
use criterion::Criterion;
use lace::cc::config::StateUpdateConfig;
use lace::cc::state::Builder;
use lace::codebook::ColType;
use lace::codebook::ValueMap;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

macro_rules! state_type_bench {
    ($id: expr, $fn: ident, $config: expr) => {
        fn $fn(c: &mut Criterion) {
            c.bench_function($id, |b| {
                let builder = Builder::new()
                    .n_rows(20)
                    .column_configs(2, $config)
                    .n_views(1)
                    .n_cats(5)
                    .seed_from_u64(1337);
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
    };
}

state_type_bench!(
    "all-categorical(2) state 20-by-2 (1 views, 5 cats)",
    bench_categorical_state,
    ColType::Categorical {
        k: 2,
        prior: None,
        hyper: None,
        value_map: ValueMap::UInt(2),
    }
);

state_type_bench!(
    "all-gaussian state 20-by-2 (1 views, 5 cats)",
    bench_gaussian_state,
    ColType::Continuous {
        hyper: None,
        prior: None
    }
);

state_type_bench!(
    "all-count-state 20-by-2 (1 views, 5 cats)",
    bench_count_state,
    ColType::Count {
        hyper: None,
        prior: None
    }
);

criterion_group!(
    state_type_benches,
    bench_categorical_state,
    bench_gaussian_state,
    bench_count_state
);
criterion_main!(state_type_benches);

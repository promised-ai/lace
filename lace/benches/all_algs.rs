use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BatchSize;
use criterion::Criterion;
use lace::cc::alg::ColAssignAlg;
use lace::cc::alg::RowAssignAlg;
use lace::cc::config::StateUpdateConfig;
use lace::cc::state::Builder;
use lace::cc::transition::StateTransition;
use lace::codebook::ColType;
use lace::codebook::ValueMap;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

const NCOLS: usize = 100;
const NROWS: usize = 1000;
const NVIEWS: usize = 10;
const NCATS: usize = 10;

macro_rules! state_type_bench {
    ($id: expr, $fn: ident, $row_alg: expr, $col_alg: expr) => {
        fn $fn(c: &mut Criterion) {
            c.bench_function($id, |b| {
                let builder = Builder::new()
                    .n_rows(NROWS)
                    .column_configs(
                        NCOLS,
                        ColType::Categorical {
                            k: 3,
                            hyper: None,
                            prior: None,
                            value_map: ValueMap::UInt(3),
                        },
                    )
                    .n_views(NVIEWS)
                    .n_cats(NCATS);

                let mut rng = Xoshiro256Plus::from_os_rng();

                b.iter_batched(
                    || builder.clone().build().unwrap(),
                    |mut state| {
                        let config = StateUpdateConfig {
                            n_iters: 1,
                            transitions: vec![
                                StateTransition::ColumnAssignment($col_alg),
                                StateTransition::StatePriorProcessParams,
                                StateTransition::RowAssignment($row_alg),
                                StateTransition::ViewPriorProcessParams,
                                StateTransition::FeaturePriors,
                            ],
                        };
                        black_box(state.update(config, &mut rng));
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    };
}

state_type_bench!(
    "genomics-data-finite-finite",
    bench_genomics_finite_finite,
    RowAssignAlg::FiniteCpu,
    ColAssignAlg::FiniteCpu
);

state_type_bench!(
    "genomics-data-slice-finite",
    bench_genomics_slice_finite,
    RowAssignAlg::Slice,
    ColAssignAlg::FiniteCpu
);

state_type_bench!(
    "genomics-data-gibbs-finite",
    bench_genomics_gibbs_finite,
    RowAssignAlg::Gibbs,
    ColAssignAlg::FiniteCpu
);

state_type_bench!(
    "genomics-data-sams-finite",
    bench_genomics_sams_finite,
    RowAssignAlg::Gibbs,
    ColAssignAlg::FiniteCpu
);

criterion_group!(
    alg_benches,
    bench_genomics_finite_finite,
    bench_genomics_slice_finite,
    bench_genomics_gibbs_finite,
    bench_genomics_sams_finite
);
criterion_main!(alg_benches);

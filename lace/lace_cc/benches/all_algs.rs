use criterion::BatchSize;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use lace_cc::alg::{ColAssignAlg, RowAssignAlg};
use lace_cc::config::StateUpdateConfig;
use lace_cc::state::Builder;
use lace_cc::transition::StateTransition;
use lace_codebook::ColType;

const NCOLS: usize = 100;
const NROWS: usize = 1000;
const NVIEWS: usize = 10;
const NCATS: usize = 10;

macro_rules! state_type_bench {
    ($id: expr, $fn: ident, $row_alg: expr, $col_alg: expr) => {
        fn $fn(c: &mut Criterion) {
            c.bench_function($id, |b| {
                let builder = StateBuilder::new()
                    .with_rows(NROWS)
                    .add_column_configs(
                        NCOLS,
                        ColType::Categorical {
                            k: 3,
                            hyper: None,
                            prior: None,
                            value_map: None,
                        },
                    )
                    .with_views(NVIEWS)
                    .with_cats(NCATS);

                let mut rng = Xoshiro256Plus::from_entropy();

                b.iter_batched(
                    || builder.clone().build().unwrap(),
                    |mut state| {
                        let config = StateUpdateConfig {
                            n_iters: 1,
                            transitions: vec![
                                StateTransition::ColumnAssignment($col_alg),
                                StateTransition::StateAlpha,
                                StateTransition::RowAssignment($row_alg),
                                StateTransition::ViewAlphas,
                                StateTransition::FeaturePriors,
                            ],
                            ..Default::default()
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

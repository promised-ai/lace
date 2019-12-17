use criterion::BatchSize;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use braid::benchmark::StateBuilder;
use braid::{Engine, InsertMode, InsertOverwrite, Row, Value};
use braid_codebook::{
    Codebook, ColMetadata, ColMetadataList, ColType, SpecType,
};
use braid_stats::Datum;

// build a one-state Engine
fn build_engine(nrows: usize, ncols: usize) -> Engine {
    let coltype = ColType::Continuous { hyper: None };

    // lightly structured view
    let builder = StateBuilder::new()
        .with_rows(nrows)
        .add_column_configs(ncols, coltype.clone())
        .with_views(2)
        .with_cats(2)
        .with_seed(1337);

    let state = builder.build().unwrap();

    let col_metadata = ColMetadataList::new(
        (0..ncols)
            .map(|id| ColMetadata {
                name: format!("{}", id),
                coltype: coltype.clone(),
                spec_type: SpecType::Other,
                notes: None,
            })
            .collect(),
    )
    .unwrap();

    let codebook = Codebook {
        table_name: "table".into(),
        state_alpha_prior: None,
        view_alpha_prior: None,
        col_metadata,
        comments: None,
        row_names: Some((0..nrows).map(|ix| format!("{}", ix)).collect()),
    };

    Engine {
        states: vec![state],
        state_ids: vec![0],
        codebook,
        rng: Xoshiro256Plus::seed_from_u64(1337),
    }
}

fn build_rows(nrows: usize, ncols: usize) -> Vec<Row> {
    let mut rng = rand::thread_rng();
    (0..nrows)
        .map(|row_ix| Row {
            row_name: format!("{}", row_ix),
            values: (0..ncols)
                .map(|col_ix| Value {
                    col_name: format!("{}", col_ix),
                    value: Datum::Continuous(rng.gen()),
                })
                .collect(),
        })
        .collect()
}

fn bench_overwrite_only(c: &mut Criterion) {
    c.bench_function("overwrite only", |b| {
        let engine = build_engine(100, 5);
        let rows = build_rows(5, 4);
        b.iter_batched(
            || (engine.clone(), rows.clone()),
            |(mut engine, rows)| {
                let mode =
                    InsertMode::DenyNewRowsAndColumns(InsertOverwrite::Allow);
                black_box(engine.insert_data(rows, None, mode));
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(insert_data_benches, bench_overwrite_only,);
criterion_main!(insert_data_benches);

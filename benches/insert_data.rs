use std::convert::TryInto;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use braid::{Engine, Row, Value, WriteMode};
use braid_cc::state::Builder;
use braid_codebook::{Codebook, ColMetadata, ColMetadataList, ColType};
use braid_data::Datum;

// build a one-state Engine
fn build_engine(nrows: usize, ncols: usize) -> Engine {
    let coltype = ColType::Continuous {
        hyper: None,
        prior: None,
    };

    // lightly structured view
    let builder = Builder::new()
        .n_rows(nrows)
        .column_configs(ncols, coltype.clone())
        .n_views(2)
        .n_cats(2)
        .seed_from_u64(1337);

    let state = builder.build().unwrap();

    let col_metadata = ColMetadataList::new(
        (0..ncols)
            .map(|id| ColMetadata {
                name: format!("{}", id),
                coltype: coltype.clone(),
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
        row_names: (0..nrows)
            .map(|ix| format!("{}", ix))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
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
            row_ix: format!("{}", row_ix).into(),
            values: (0..ncols)
                .map(|col_ix| Value {
                    col_ix: format!("{}", col_ix).into(),
                    value: Datum::Continuous(rng.gen()),
                })
                .collect(),
        })
        .collect()
}

fn bench_overwrite_only(c: &mut Criterion) {
    c.bench_function("insert_data overwrite only", |b| {
        let engine = build_engine(100, 5);
        let rows = build_rows(5, 4);
        b.iter_batched(
            || (engine.clone(), rows.clone()),
            |(mut engine, rows)| {
                let mode = WriteMode::unrestricted();
                black_box(engine.insert_data(rows, None, None, mode).unwrap());
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_append_rows(c: &mut Criterion) {
    c.bench_function("insert_data append rows", |b| {
        let engine = build_engine(100, 5);

        let rows: Vec<Row> = build_rows(5, 4)
            .drain(..)
            .enumerate()
            .map(|(ix, mut row)| {
                row.row_ix = format!("{}", 100 + ix).into();
                row
            })
            .collect();

        b.iter_batched(
            || (engine.clone(), rows.clone()),
            |(mut engine, rows)| {
                let mode = WriteMode::unrestricted();
                black_box(engine.insert_data(rows, None, None, mode).unwrap());
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(insert_data_benches, bench_overwrite_only, bench_append_rows);
criterion_main!(insert_data_benches);

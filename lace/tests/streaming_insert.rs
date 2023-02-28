use std::convert::TryInto;

use lace::{AppendStrategy, Engine, HasData, HasStates, WriteMode};
use lace_cc::state::Builder;
use lace_codebook::{Codebook, ColMetadata, ColType};
use lace_data::Datum;
use lace_stats::prior::nix::NixHyper;

use lace_stats::rv::dist::Gamma;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

fn assert_rows_eq(row_a: &[Datum], row_b: &[Datum]) {
    assert_eq!(row_a.len(), row_b.len());
    for (ix, (a, b)) in row_a.iter().zip(row_b.iter()).enumerate() {
        let xa = a.to_f64_opt().unwrap();
        let xb = b.to_f64_opt().unwrap();
        if (xa - xb).abs() > 1E-14 {
            let msg = format!(
                "Rows were different at index {}: {:?} != {:?}",
                ix, a, b
            );
            panic!("{}\n{:?} != {:?}", msg, row_a, row_b);
        }
    }
}

fn assert_rows_ne(row_a: &[Datum], row_b: &[Datum]) {
    assert_eq!(row_a.len(), row_b.len());
    let diff = row_a.iter().zip(row_b.iter()).fold(false, |acc, (a, b)| {
        if acc {
            acc
        } else {
            let xa = a.to_f64_opt().unwrap();
            let xb = b.to_f64_opt().unwrap();
            (xa - xb).abs() > 1E-14
        }
    });

    if !diff {
        panic!("Rows identical\n{:?} == {:?}", row_a, row_b);
    }
}

fn gen_engine() -> Engine {
    let states: Vec<_> = (0..4)
        .map(|_| {
            Builder::new()
                .n_rows(10)
                .column_configs(
                    14,
                    ColType::Continuous {
                        hyper: Some(NixHyper::default()),
                        prior: None,
                    },
                )
                .n_views(1)
                .n_cats(2)
                .build()
                .unwrap()
        })
        .collect();

    let codebook = Codebook {
        table_name: "table".into(),
        state_alpha_prior: Some(Gamma::default().into()),
        view_alpha_prior: Some(Gamma::default().into()),
        col_metadata: (0..14)
            .map(|i| ColMetadata {
                name: format!("{}", i),
                notes: None,
                coltype: ColType::Continuous {
                    hyper: Some(NixHyper::default()),
                    prior: None,
                },
                missing_not_at_random: false,
            })
            .collect::<Vec<ColMetadata>>()
            .try_into()
            .unwrap(),
        comments: None,
        row_names: (0..10)
            .map(|i| format!("{}", i))
            .collect::<Vec<String>>()
            .try_into()
            .unwrap(),
    };

    Engine {
        states,
        state_ids: vec![0, 1, 2, 3],
        rng: Xoshiro256Plus::from_entropy(),
        codebook,
    }
}

#[test]
fn stream_insert_all_data() {
    let mut engine = gen_engine();

    let mut rng = rand::thread_rng();

    let mode = WriteMode {
        append_strategy: AppendStrategy::Window,
        ..WriteMode::unrestricted()
    };

    for i in 10..40 {
        let row = (
            format!("{}", i),
            (0..14)
                .map(|j| {
                    let x = Datum::Continuous(rng.gen());
                    (format!("{}", j), x)
                })
                .collect::<Vec<(String, Datum)>>(),
        );
        let tasks = engine
            .insert_data(vec![row.into()], None, None, mode)
            .unwrap();
        assert_eq!(tasks.new_rows().unwrap().len(), 1);
        engine.run(1).unwrap();
        assert_eq!(engine.n_rows(), 10);
    }
}

#[test]
fn trench_insert_all_data() {
    let mut engine = gen_engine();

    let mut rng = rand::thread_rng();

    let mode = WriteMode {
        append_strategy: AppendStrategy::Trench {
            max_n_rows: 15,
            trench_ix: 10,
        },
        ..WriteMode::unrestricted()
    };

    let ninth_row: Vec<_> =
        (0..14).map(|col_ix| engine.cell(9, col_ix)).collect();

    let mut last_tenth_row: Vec<_> =
        (0..14).map(|col_ix| engine.cell(9, col_ix)).collect();

    for (i, ix) in (10..40).enumerate() {
        let row = (
            format!("{}", ix),
            (0..14)
                .map(|j| {
                    let x = Datum::Continuous(rng.gen());
                    (format!("{}", j), x)
                })
                .collect::<Vec<(String, Datum)>>(),
        );
        let tasks = engine
            .insert_data(vec![row.into()], None, None, mode)
            .unwrap();

        let this_ninth_row: Vec<_> =
            (0..14).map(|col_ix| engine.cell(9, col_ix)).collect();

        let this_tenth_row: Vec<_> =
            (0..14).map(|col_ix| engine.cell(10, col_ix)).collect();

        engine.run(1).unwrap();

        dbg!(i);
        assert_eq!(tasks.new_rows().unwrap().len(), 1);
        assert_eq!(engine.n_rows(), 15_usize.min(10 + i + 1));

        assert_rows_eq(&ninth_row, &this_ninth_row);
        if ix > 14 {
            assert_rows_ne(&last_tenth_row, &this_tenth_row);
        }

        last_tenth_row = this_tenth_row;
    }
}

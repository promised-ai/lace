use std::convert::Into;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use braid::cc::config::EngineUpdateConfig;
use braid::data::DataSource;
use braid::examples::Example;
use braid::{
    AppendStrategy, Engine, EngineBuilder, InsertDataActions, SupportExtension,
};
use braid_codebook::Codebook;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn animals_data_path() -> PathBuf {
    Path::new("resources")
        .join("datasets")
        .join("animals")
        .join("data.csv")
}

fn animals_codebook_path() -> PathBuf {
    Path::new("resources")
        .join("datasets")
        .join("animals")
        .join("codebook.yaml")
}

// TODO: Don't use tiny test files, generate them in code from raw strings and
// tempfiles.
fn engine_from_csv<P: Into<PathBuf>>(path: P) -> Engine {
    EngineBuilder::new(DataSource::Csv(path.into()))
        .with_nstates(2)
        .build()
        .unwrap()
}

#[test]
fn loaded_engine_should_have_same_rng_state() {
    {
        // Make sure the engine loads from a file. If the Animals example does
        // not exist already, the example will be run and directly converted to
        // an engine. We need to run it at least once so the metadata is saved,
        // then the subsequent engines will come from that same serialized
        // metadata.
        let _engine = Example::Animals.engine().unwrap();
    }
    let mut engine_1 = Example::Animals.engine().unwrap();
    let mut engine_2 = Example::Animals.engine().unwrap();
    engine_1.run(5);
    engine_2.run(5);

    for (s1, s2) in engine_1.states.iter().zip(engine_2.states.iter()) {
        assert_eq!(s1.asgn.asgn, s2.asgn.asgn);
    }
}

#[test]
fn zero_states_to_new_causes_error() {
    let codebook = {
        let mut file = File::open(animals_codebook_path()).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        serde_yaml::from_slice(data.as_bytes()).unwrap()
    };
    let rng = Xoshiro256Plus::from_entropy();
    match Engine::new(
        0,
        codebook,
        DataSource::Csv(animals_data_path().into()),
        0,
        rng,
    ) {
        Err(braid::error::NewEngineError::ZeroStatesRequested) => (),
        Err(_) => panic!("wrong error"),
        Ok(_) => panic!("Failed to catch zero states error"),
    }
}

#[test]
fn save_run_load_run_should_add_iterations() {
    let dir = tempfile::TempDir::new().unwrap();

    {
        let mut engine = engine_from_csv("resources/test/small/small.csv");

        engine.run(100);

        for state in engine.states.iter() {
            assert_eq!(state.diagnostics.loglike.len(), 100);
            assert_eq!(state.diagnostics.nviews.len(), 100);
            assert_eq!(state.diagnostics.state_alpha.len(), 100);
        }

        engine.save_to(dir.as_ref()).save().unwrap();
    }

    {
        let mut engine = Engine::load(dir.as_ref()).unwrap();

        for state in engine.states.iter() {
            assert_eq!(state.diagnostics.loglike.len(), 100);
            assert_eq!(state.diagnostics.nviews.len(), 100);
            assert_eq!(state.diagnostics.state_alpha.len(), 100);
        }

        engine.run(10);

        for state in engine.states.iter() {
            assert_eq!(state.diagnostics.loglike.len(), 110);
            assert_eq!(state.diagnostics.nviews.len(), 110);
            assert_eq!(state.diagnostics.state_alpha.len(), 110);
        }
    }
}

#[test]
fn run_empty_engine_smoke_test() {
    let mut engine = Engine::new(
        1,
        Codebook::default(),
        DataSource::Empty,
        0,
        Xoshiro256Plus::seed_from_u64(0xABCD),
    )
    .unwrap();

    engine.run(100)
}

#[test]
fn update_empty_engine_smoke_test() {
    let mut engine = Engine::new(
        1,
        Codebook::default(),
        DataSource::Empty,
        0,
        Xoshiro256Plus::seed_from_u64(0xABCD),
    )
    .unwrap();

    engine.update(EngineUpdateConfig::default());
}

#[test]
fn run_engine_after_flatten_cols_smoke_test() {
    let mut engine = Example::Satellites.engine().unwrap();
    assert!(engine.states.iter().any(|state| state.nviews() > 1));
    engine.flatten_cols();
    assert!(engine.states.iter().all(|state| state.nviews() == 1));
    engine.run(1);
}

mod contructor {
    use super::*;
    use braid::error::{DataParseError, NewEngineError};
    use braid_codebook::{ColMetadata, ColType};
    use std::convert::TryInto;

    #[test]
    fn non_empty_col_metadata_empty_data_source_errors() {
        let err = Engine::new(
            1,
            Codebook {
                col_metadata: vec![ColMetadata {
                    name: String::from("one"),
                    coltype: ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                    notes: None,
                }]
                .try_into()
                .unwrap(),
                ..Default::default()
            },
            DataSource::Empty,
            0,
            Xoshiro256Plus::seed_from_u64(0xABCD),
        )
        .unwrap_err();

        match err {
            NewEngineError::DataParseError(
                DataParseError::ColumnMetadataSuppliedForEmptyData,
            ) => (),
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn non_empty_row_names_empty_data_source_errors() {
        let err = Engine::new(
            1,
            Codebook {
                row_names: vec![String::from("one")].try_into().unwrap(),
                ..Default::default()
            },
            DataSource::Empty,
            0,
            Xoshiro256Plus::seed_from_u64(0xABCD),
        )
        .unwrap_err();

        match err {
            NewEngineError::DataParseError(
                DataParseError::RowNamesSuppliedForEmptyData,
            ) => (),
            _ => panic!("wrong error"),
        }
    }
}

#[test]
fn cell_gibbs_smoke() {
    let mut engine = Example::Animals.engine().unwrap();
    for _ in 0..100 {
        engine.cell_gibbs(0, 0);
    }
    for _ in 0..100 {
        engine.cell_gibbs(15, 12);
    }
}

#[cfg(test)]
mod prior_in_codebook {
    use super::*;
    use braid::cc::ColModel;
    use braid_codebook::{Codebook, ColMetadata, ColMetadataList, ColType};
    use braid_stats::prior::crp::CrpPrior;
    use braid_stats::prior::ng::NgHyper;
    use rv::dist::{Gamma, NormalGamma};
    use rv::traits::Rv;
    use std::convert::TryInto;
    use std::io::Write;

    // Generate a two-column codebook ('x' and 'y'). The x column will alyways
    // have a hyper for the x column, but will have a prior defined if set_prior
    // is true. The y column will have neither a prior or hyper defined.
    fn gen_codebook(nrows: usize, set_prior: bool) -> Codebook {
        Codebook {
            table_name: String::from("table"),
            state_alpha_prior: Some(CrpPrior::Gamma(Gamma::default())),
            view_alpha_prior: Some(CrpPrior::Gamma(Gamma::default())),
            col_metadata: {
                let mut col_metadata = ColMetadataList::new(vec![]).unwrap();
                col_metadata
                    .push(ColMetadata {
                        name: String::from("x"),
                        notes: None,
                        coltype: ColType::Continuous {
                            hyper: Some(NgHyper::default()),
                            prior: if set_prior {
                                Some(NormalGamma::new_unchecked(
                                    0.0, 1.0, 2.0, 3.0,
                                ))
                            } else {
                                None
                            },
                        },
                    })
                    .unwrap();

                col_metadata
                    .push(ColMetadata {
                        name: String::from("y"),
                        notes: None,
                        coltype: ColType::Continuous {
                            hyper: None,
                            prior: None,
                        },
                    })
                    .unwrap();
                col_metadata
            },
            row_names: (0..nrows)
                .map(|i| format!("{}", i))
                .collect::<Vec<String>>()
                .try_into()
                .unwrap(),
            comments: None,
        }
    }

    fn gen_codebook_text(nrows: usize) -> Codebook {
        use indoc::indoc;
        let mut text = indoc!(
            "
        ---
        table_name: table
        state_alpha_prior:
            Gamma:
                shape: 1.0
                rate: 1.0
        view_alpha_prior:
            Gamma:
                shape: 1.0
                rate: 1.0
        col_metadata:
            - name: x
              coltype:
                Continuous:
                    prior:
                        m: 0.0
                        r: 1.0
                        s: 2.0
                        v: 3.0
            - name: y
              coltype:
                Continuous:
                    hyper: ~
                    prior: ~
        comments: ~
        row_names:
        "
        )
        .to_string();

        for i in 0..nrows {
            text = text + &format!("  - {}\n", i);
        }

        serde_yaml::from_str(&text).unwrap()
    }

    fn get_prior_ref(engine: &Engine, col_ix: usize) -> &NormalGamma {
        match engine.states[0].feature(col_ix) {
            ColModel::Continuous(col) => &col.prior,
            _ => panic!("unexpected ColModel variant"),
        }
    }

    fn get_prior_params(
        engine: &Engine,
        col_ix: usize,
    ) -> (f64, f64, f64, f64) {
        let ng = get_prior_ref(engine, col_ix);
        (ng.m(), ng.r(), ng.s(), ng.v())
    }

    fn run_test(nrows: usize, codebook: Codebook) {
        let mut csvfile = tempfile::NamedTempFile::new().unwrap();
        let mut rng = Xoshiro256Plus::from_entropy();
        let gauss = rv::dist::Gaussian::standard();

        write!(csvfile, "id,x,y\n").unwrap();
        for i in 0..nrows {
            let x: f64 = gauss.draw(&mut rng);
            let y: f64 = gauss.draw(&mut rng);
            write!(csvfile, "{},{},{}", i, x, y).unwrap();
            if i < 99 {
                write!(csvfile, "\n").unwrap();
            }
        }

        let mut engine = Engine::new(
            1,
            codebook,
            DataSource::Csv(csvfile.path().to_path_buf()),
            0,
            rng,
        )
        .unwrap();

        let target_params = (0.0, 1.0, 2.0, 3.0);
        let x_start_params = get_prior_params(&engine, 0);
        assert_eq!(x_start_params, target_params);

        let mut last_y_params = get_prior_params(&engine, 1);
        for _ in 0..5 {
            engine.run(5);
            let x_params = get_prior_params(&engine, 0);
            let current_y_params = get_prior_params(&engine, 1);

            assert_eq!(x_params, target_params);
            assert_ne!(last_y_params, current_y_params);
            last_y_params = current_y_params;
        }
    }

    #[test]
    fn setting_prior_in_codebook_struct_disables_prior_updates_with_csv_data() {
        let nrows = 100;
        let codebook = gen_codebook(nrows, true);
        run_test(nrows, codebook)
    }

    #[test]
    fn setting_prior_in_codebook_yaml_disables_prior_updates_with_csv_data() {
        let nrows = 100;
        let codebook = gen_codebook_text(nrows);
        run_test(nrows, codebook)
    }
}

// NOTE: These tests make sure that values have been updated, that the desired
// rows and columns have been added, and that bad inputs return the correct
// errors. They do not make sure the State metadata (assignment and sufficient
// statistics) have been updated properly. Those tests occur in State.
mod insert_data {
    use super::*;
    use braid::cc::{ColAssignAlg, FType, RowAssignAlg, StateTransition};
    use braid::error::InsertDataError;
    use braid::examples::animals;
    use braid::{InsertMode, OracleT, OverwriteMode, Row, Value, WriteMode};
    use braid_codebook::{ColMetadata, ColMetadataList, ColType};
    use braid_stats::prior::csd::CsdHyper;
    use braid_stats::Datum;
    use maplit::{btreemap, hashmap};

    #[test]
    fn add_new_row_to_animals_adds_values_in_empty_row() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();
        let starting_cols = engine.ncols();

        let rows = vec![Row {
            row_name: "pegasus".into(),
            values: vec![
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "hooves".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(0),
                },
            ],
        }];

        let actions = engine
            .insert_data(
                rows,
                None,
                None,
                WriteMode {
                    insert: InsertMode::DenyNewColumns,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        assert_eq!(engine.nrows(), starting_rows + 1);
        assert_eq!(engine.ncols(), starting_cols);
        assert!(actions.support_extensions().is_none());
        assert!(actions.new_cols().is_none());
        assert!(actions.new_rows().is_some());
        assert!(actions.new_rows().unwrap().contains("pegasus"));

        let row_ix = starting_rows;

        for col_ix in 0..engine.ncols() {
            let datum = engine.datum(row_ix, col_ix).unwrap();
            match col_ix {
                // hooves
                20 => assert_eq!(datum, Datum::Categorical(1)),
                // flys
                34 => assert_eq!(datum, Datum::Categorical(1)),
                // swims
                36 => assert_eq!(datum, Datum::Categorical(0)),
                _ => assert_eq!(datum, Datum::Missing),
            }
        }
    }

    #[test]
    fn add_new_row_after_new_row_adds_two_rows() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        {
            let rows = vec![Row {
                row_name: "pegasus".into(),
                values: vec![Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                }],
            }];

            let actions = engine
                .insert_data(
                    rows,
                    None,
                    None,
                    WriteMode {
                        insert: InsertMode::DenyNewColumns,
                        overwrite: OverwriteMode::Deny,
                        allow_extend_support: false,
                        append_strategy: AppendStrategy::None,
                    },
                )
                .unwrap();

            assert_eq!(engine.nrows(), starting_rows + 1);
            assert!(actions.support_extensions().is_none());
            assert!(actions.new_cols().is_none());
            assert!(actions.new_rows().is_some());
            assert!(actions.new_rows().unwrap().contains("pegasus"));
        }

        {
            let rows = vec![Row {
                row_name: "yoshi".into(),
                values: vec![Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(0),
                }],
            }];

            let actions = engine
                .insert_data(
                    rows,
                    None,
                    None,
                    WriteMode {
                        insert: InsertMode::DenyNewColumns,
                        overwrite: OverwriteMode::Deny,
                        allow_extend_support: false,
                        append_strategy: AppendStrategy::None,
                    },
                )
                .unwrap();

            assert_eq!(engine.nrows(), starting_rows + 2);
            assert!(actions.support_extensions().is_none());
            assert!(actions.new_cols().is_none());
            assert!(actions.new_rows().is_some());
            assert!(actions.new_rows().unwrap().contains("yoshi"));
        }
    }

    #[test]
    fn readd_new_row_after_new_row_adds_one_row() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        {
            let rows = vec![Row {
                row_name: "pegasus".into(),
                values: vec![Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                }],
            }];

            let actions = engine
                .insert_data(
                    rows,
                    None,
                    None,
                    WriteMode {
                        insert: InsertMode::DenyNewColumns,
                        overwrite: OverwriteMode::Deny,
                        allow_extend_support: false,
                        append_strategy: AppendStrategy::None,
                    },
                )
                .unwrap();

            assert_eq!(engine.nrows(), starting_rows + 1);
            assert!(actions.support_extensions().is_none());
            assert!(actions.new_cols().is_none());
            assert!(actions.new_rows().is_some());
            assert!(actions.new_rows().unwrap().contains("pegasus"));
        }

        {
            let rows = vec![Row {
                row_name: "pegasus".into(),
                values: vec![Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(0),
                }],
            }];

            let actions = engine
                .insert_data(
                    rows,
                    None,
                    None,
                    WriteMode {
                        insert: InsertMode::DenyNewRowsAndColumns,
                        overwrite: OverwriteMode::MissingOnly,
                        allow_extend_support: false,
                        append_strategy: AppendStrategy::None,
                    },
                )
                .unwrap();

            assert_eq!(engine.nrows(), starting_rows + 1);
            assert!(actions.support_extensions().is_none());
            assert!(actions.new_cols().is_none());
            assert!(actions.new_rows().is_none());
        }
    }

    #[test]
    fn update_value_replaces_value() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();
        let starting_cols = engine.ncols();

        let rows = vec![Row {
            row_name: "bat".into(),
            values: vec![Value {
                col_name: "flys".into(),
                value: Datum::Categorical(0),
            }],
        }];

        assert_eq!(engine.datum(29, 34).unwrap(), Datum::Categorical(1));

        let actions = engine
            .insert_data(
                rows,
                None,
                None,
                WriteMode {
                    insert: InsertMode::DenyNewRowsAndColumns,
                    overwrite: OverwriteMode::Allow,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        assert!(actions.support_extensions().is_none());
        assert!(actions.new_cols().is_none());
        assert!(actions.new_rows().is_none());
        assert_eq!(engine.nrows(), starting_rows);
        assert_eq!(engine.ncols(), starting_cols);

        assert_eq!(engine.datum(29, 34).unwrap(), Datum::Categorical(0));
    }

    #[test]
    fn insert_missing_removes_value() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();
        let starting_cols = engine.ncols();

        let rows = vec![Row {
            row_name: "bat".into(),
            values: vec![Value {
                col_name: "flys".into(),
                value: Datum::Missing,
            }],
        }];

        assert_eq!(engine.datum(29, 34).unwrap(), Datum::Categorical(1));

        let actions = engine
            .insert_data(
                rows,
                None,
                None,
                WriteMode {
                    insert: InsertMode::DenyNewRowsAndColumns,
                    overwrite: OverwriteMode::Allow,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        assert!(actions.support_extensions().is_none());
        assert!(actions.new_cols().is_none());
        assert!(actions.new_rows().is_none());
        assert_eq!(engine.nrows(), starting_rows);
        assert_eq!(engine.ncols(), starting_cols);

        assert_eq!(engine.datum(29, 34).unwrap(), Datum::Missing)
    }

    #[test]
    fn insert_value_into_new_col_existing_row_creates_col() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        let rows = vec![Row {
            row_name: "bat".into(),
            values: vec![Value {
                col_name: "sucks+blood".into(),
                value: Datum::Categorical(1),
            }],
        }];

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "sucks+blood".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: Some(CsdHyper::default()),
                value_map: None,
                prior: None,
            },
            notes: None,
        }])
        .unwrap();

        assert_eq!(engine.ncols(), 85);

        let actions = engine
            .insert_data(
                rows,
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::DenyNewRows,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::Window,
                },
            )
            .unwrap();

        assert_eq!(engine.nrows(), starting_rows);
        assert_eq!(engine.ncols(), 86);
        assert!(actions.support_extensions().is_none());
        assert!(actions.new_rows().is_none());
        assert!(actions.new_cols().is_some());
        assert!(actions.new_cols().unwrap().contains("sucks+blood"));

        for row_ix in 0..engine.nrows() {
            let datum = engine.datum(row_ix, 85).unwrap();
            if row_ix == 29 {
                assert_eq!(datum, Datum::Categorical(1));
            } else {
                assert_eq!(datum, Datum::Missing);
            }
        }
    }

    #[test]
    fn insert_value_into_new_col_existing_row_wrong_datum_type_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "sucks+blood".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
                prior: None,
            },
            notes: None,
        }])
        .unwrap();

        let rows = vec![Row {
            row_name: "bat".into(),
            values: vec![Value {
                col_name: "sucks+blood".into(),
                value: Datum::Continuous(1.0), // should be categorical
            }],
        }];

        assert_eq!(engine.ncols(), 85);

        let err = engine
            .insert_data(
                rows,
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::DenyNewRows,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap_err();

        assert_eq!(
            err,
            InsertDataError::DatumIncompatibleWithColumn {
                col: String::from("sucks+blood"),
                ftype: FType::Categorical,
                ftype_req: FType::Continuous,
            }
        )
    }

    #[test]
    fn insert_value_into_new_col_in_new_row_creates_new_row_and_col() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "vampire".into(),
            values: vec![Value {
                col_name: "sucks+blood".into(),
                value: Datum::Categorical(1),
            }],
        }];

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "sucks+blood".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: Some(CsdHyper::default()),
                prior: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        assert_eq!(engine.ncols(), 85);

        let actions = engine
            .insert_data(
                rows,
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::Unrestricted,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        assert_eq!(engine.nrows(), 51);
        assert_eq!(engine.ncols(), 86);
        assert!(actions.support_extensions().is_none());
        assert!(actions.new_rows().unwrap().contains("vampire"));
        assert!(actions.new_cols().unwrap().contains("sucks+blood"));

        for row_ix in 0..engine.nrows() {
            let datum = engine.datum(row_ix, 85).unwrap();
            if row_ix == 50 {
                assert_eq!(datum, Datum::Categorical(1));
            } else {
                assert_eq!(datum, Datum::Missing);
            }
        }

        for col_ix in 0..engine.ncols() {
            let datum = engine.datum(50, col_ix).unwrap();
            if col_ix == 85 {
                assert_eq!(datum, Datum::Categorical(1));
            } else {
                assert_eq!(datum, Datum::Missing);
            }
        }
    }

    #[test]
    fn overwrite_when_deny_raises_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "bat".into(),
            values: vec![Value {
                col_name: "flys".into(),
                value: Datum::Categorical(0),
            }],
        }];

        assert_eq!(engine.datum(29, 34).unwrap(), Datum::Categorical(1));

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::Deny,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InsertDataError::ModeForbidsOverwrite);
    }

    #[test]
    fn overwrite_when_missing_only_raises_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "bat".into(),
            values: vec![Value {
                col_name: "flys".into(),
                value: Datum::Categorical(0),
            }],
        }];

        assert_eq!(engine.datum(29, 34).unwrap(), Datum::Categorical(1));

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::MissingOnly,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InsertDataError::ModeForbidsOverwrite);
    }

    #[test]
    fn insert_value_into_new_col_in_new_row_when_new_cols_denied_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "vampire".into(),
            values: vec![Value {
                col_name: "sucks+blood".into(),
                value: Datum::Categorical(1),
            }],
        }];

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "sucks+blood".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
                prior: None,
            },
            notes: None,
        }])
        .unwrap();

        let result = engine.insert_data(
            rows,
            Some(col_metadata),
            None,
            WriteMode {
                insert: InsertMode::DenyNewColumns,
                overwrite: OverwriteMode::Deny,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InsertDataError::ModeForbidsNewColumns);
    }

    #[test]
    fn insert_value_into_new_row_in_new_row_when_new_row_denied_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "vampire".into(),
            values: vec![Value {
                col_name: "sucks+blood".into(),
                value: Datum::Categorical(1),
            }],
        }];

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "sucks+blood".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                prior: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        let result = engine.insert_data(
            rows,
            Some(col_metadata),
            None,
            WriteMode {
                insert: InsertMode::DenyNewRows,
                overwrite: OverwriteMode::Deny,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InsertDataError::ModeForbidsNewRows);
    }

    #[test]
    fn insert_value_into_new_rows_when_new_rows_disallowed_error() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "vampire".into(),
            values: vec![Value {
                col_name: "flys".into(),
                value: Datum::Missing,
            }],
        }];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRows,
                overwrite: OverwriteMode::Allow,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InsertDataError::ModeForbidsNewRows);
    }

    #[test]
    fn insert_value_into_new_col_in_new_row_runs_after() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "vampire".into(),
            values: vec![Value {
                col_name: "sucks+blood".into(),
                value: Datum::Categorical(1),
            }],
        }];

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "sucks+blood".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: Some(CsdHyper::default()),
                prior: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        engine
            .insert_data(
                rows,
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::Unrestricted,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        assert_eq!(engine.nrows(), 51);
        assert_eq!(engine.ncols(), 86);

        engine.run(5);

        assert_eq!(engine.nrows(), 51);
        assert_eq!(engine.ncols(), 86);
    }

    #[test]
    fn insert_into_empty() {
        use braid_stats::prior::ng::NgHyper;
        use rv::dist::{Gamma, Gaussian};

        let values = vec![Value {
            col_name: "score".to_string(),
            value: Datum::Continuous((12345.0_f64).ln()),
        }];

        let row = Row {
            row_name: "1".to_string(),
            values,
        };

        let col_type = ColType::Continuous {
            hyper: Some(NgHyper {
                pr_m: Gaussian::new_unchecked(0.0, 1.0),
                pr_r: Gamma::new_unchecked(2.0, 1.0),
                pr_s: Gamma::new_unchecked(1.0, 1.0),
                pr_v: Gamma::new_unchecked(2.0, 1.0),
            }),
            prior: None,
        };

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "score".to_string(),
            coltype: col_type.clone(),
            notes: None,
        }])
        .unwrap();

        let mut engine = Engine::new(
            1,
            Codebook::default(),
            DataSource::Empty,
            0,
            Xoshiro256Plus::seed_from_u64(0xABCD),
        )
        .unwrap();

        let actions = engine
            .insert_data(
                vec![row],
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::Unrestricted,
                    overwrite: OverwriteMode::Allow,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .expect("Failed to insert data");

        assert_eq!(engine.nrows(), 1);
        assert_eq!(engine.ncols(), 1);

        assert!(actions.support_extensions().is_none());
        assert!(actions.new_cols().unwrap().contains("score"));
        assert!(actions.new_rows().unwrap().contains("1"));
    }

    #[test]
    fn engine_saves_inserted_data_rows() {
        let dir = tempfile::TempDir::new().unwrap();

        let mut engine = {
            let engine = Example::Animals.engine().unwrap();
            engine.save_to(dir.path()).save().unwrap();
            Engine::load(dir.path()).unwrap()
        };

        assert_eq!(engine.nrows(), 50);
        assert_eq!(engine.ncols(), 85);

        let new_row: Row = (
            "tribble",
            vec![
                ("hunter", Datum::Categorical(0)),
                ("fierce", Datum::Categorical(1)),
            ],
        )
            .into();

        engine
            .insert_data(
                vec![new_row],
                None,
                None,
                WriteMode {
                    insert: InsertMode::DenyNewColumns,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        engine.save_to(dir.path()).save().unwrap();

        let engine = Engine::load(dir.path()).unwrap();

        assert_eq!(engine.nrows(), 51);
        assert_eq!(engine.ncols(), 85);
        assert_eq!(engine.datum(50, 58).unwrap(), Datum::Categorical(0));
        assert_eq!(engine.datum(50, 78).unwrap(), Datum::Categorical(1));
        assert_eq!(engine.datum(50, 11).unwrap(), Datum::Missing);

        assert_eq!(engine.codebook.row_names[50], String::from("tribble"));
    }

    #[test]
    fn engine_saves_inserted_data_cols() {
        let dir = tempfile::TempDir::new().unwrap();

        let mut engine = {
            let engine = Example::Animals.engine().unwrap();
            engine.save_to(dir.path()).save().unwrap();
            Engine::load(dir.path()).unwrap()
        };

        assert_eq!(engine.ncols(), 85);

        let new_col: Vec<Row> = vec![
            ("pig", vec![("cuddly", Datum::Categorical(1))]).into(),
            ("wolf", vec![("cuddly", Datum::Categorical(0))]).into(),
        ];

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "cuddly".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: Some(CsdHyper::default()),
                prior: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        engine
            .insert_data(
                new_col,
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::DenyNewRows,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        engine.save_to(dir.path()).save().unwrap();

        let engine = Engine::load(dir.path()).unwrap();

        assert_eq!(engine.ncols(), 86);
        assert_eq!(engine.nrows(), 50);
        assert_eq!(engine.datum(41, 85).unwrap(), Datum::Categorical(1));
        assert_eq!(engine.datum(31, 85).unwrap(), Datum::Categorical(0));
        assert_eq!(engine.datum(32, 85).unwrap(), Datum::Missing);
        assert!(engine.codebook.col_metadata.contains_key("cuddly"));
    }

    #[test]
    fn engine_saves_inserted_data_rows_into_empty() {
        let dir = tempfile::TempDir::new().unwrap();

        let mut engine = {
            let engine = Engine::new(
                1,
                Codebook::default(),
                DataSource::Empty,
                0,
                Xoshiro256Plus::seed_from_u64(0xABCD),
            )
            .unwrap();
            engine.save_to(dir.path()).save().unwrap();
            Engine::load(dir.path()).unwrap()
        };

        assert_eq!(engine.nrows(), 0);
        assert_eq!(engine.ncols(), 0);

        let new_row: Row = (
            "tribble",
            vec![
                ("hunter", Datum::Categorical(0)),
                ("fierce", Datum::Categorical(1)),
            ],
        )
            .into();

        let col_metadata = ColMetadataList::new(vec![
            ColMetadata {
                name: "hunter".into(),
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: Some(CsdHyper::default()),
                    prior: None,
                    value_map: None,
                },
                notes: None,
            },
            ColMetadata {
                name: "fierce".into(),
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: Some(CsdHyper::default()),
                    prior: None,
                    value_map: None,
                },
                notes: None,
            },
        ])
        .unwrap();

        engine
            .insert_data(
                vec![new_row],
                Some(col_metadata),
                None,
                WriteMode {
                    insert: InsertMode::Unrestricted,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        engine.save_to(dir.path()).save().unwrap();

        let engine = Engine::load(dir.path()).unwrap();

        assert_eq!(engine.nrows(), 1);
        assert_eq!(engine.ncols(), 2);
        assert_eq!(engine.datum(0, 0).unwrap(), Datum::Categorical(0));
        assert_eq!(engine.datum(0, 1).unwrap(), Datum::Categorical(1));

        assert_eq!(engine.codebook.row_names[0], String::from("tribble"));
    }

    #[test]
    fn repeated_insert_and_update_into_empty_engine_1_col() {
        fn add_row(
            engine: &mut Engine,
            name: &str,
            x: f64,
        ) -> Result<InsertDataActions, InsertDataError> {
            use braid_stats::prior::ng::NgHyper;

            let row = Row {
                row_name: name.to_string(),
                values: vec![Value {
                    col_name: "data".to_string(),
                    value: Datum::Continuous(x),
                }],
            };
            let colmd = ColMetadata {
                name: "data".to_string(),
                notes: None,
                coltype: ColType::Continuous {
                    hyper: Some(NgHyper::default()),
                    prior: None,
                },
            };
            engine.insert_data(
                vec![row],
                Some(ColMetadataList::new(vec![colmd]).unwrap()),
                None,
                WriteMode {
                    insert: InsertMode::Unrestricted,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
        }

        let cfg = EngineUpdateConfig {
            n_iters: 10,
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::Gibbs),
                StateTransition::StateAlpha,
                StateTransition::RowAssignment(RowAssignAlg::Gibbs),
                StateTransition::ViewAlphas,
                StateTransition::FeaturePriors,
            ],
            ..Default::default()
        };

        let mut engine = EngineBuilder::new(DataSource::Empty).build().unwrap();
        assert_eq!(engine.nrows(), 0);
        assert_eq!(engine.ncols(), 0);

        add_row(&mut engine, "v1", 1.0).unwrap();
        add_row(&mut engine, "v2", -1.0).unwrap();
        add_row(&mut engine, "v3", 0.0).unwrap();
        assert_eq!(engine.nrows(), 3);
        assert_eq!(engine.ncols(), 1);

        engine.update(cfg.clone());

        add_row(&mut engine, "b1", 1.0).unwrap();

        assert_eq!(engine.nrows(), 4);
        assert_eq!(engine.ncols(), 1);
        engine.update(cfg.clone());
        assert_eq!(engine.nrows(), 4);

        add_row(&mut engine, "b2", -1.0).unwrap();

        assert_eq!(engine.nrows(), 5);
        engine.update(cfg.clone());
        assert_eq!(engine.nrows(), 5);

        add_row(&mut engine, "b3", 0.0).unwrap();

        assert_eq!(engine.nrows(), 6);
        engine.update(cfg);
        assert_eq!(engine.nrows(), 6);
    }

    #[test]
    fn repeated_insert_and_update_into_empty_engine_2_cols() {
        fn add_row(
            engine: &mut Engine,
            name: &str,
            x: f64,
            y: f64,
        ) -> Result<InsertDataActions, InsertDataError> {
            use braid_stats::prior::ng::NgHyper;

            let row = Row {
                row_name: name.to_string(),
                values: vec![
                    Value {
                        col_name: "x".to_string(),
                        value: Datum::Continuous(x),
                    },
                    Value {
                        col_name: "y".to_string(),
                        value: Datum::Continuous(y),
                    },
                ],
            };

            let colmd_x = ColMetadata {
                name: "x".into(),
                notes: None,
                coltype: ColType::Continuous {
                    hyper: Some(NgHyper::default()),
                    prior: None,
                },
            };

            let colmd_y = {
                let mut colmd = colmd_x.clone();
                colmd.name = "y".into();
                colmd
            };

            engine.insert_data(
                vec![row],
                Some(ColMetadataList::new(vec![colmd_x, colmd_y]).unwrap()),
                None,
                WriteMode {
                    insert: InsertMode::Unrestricted,
                    overwrite: OverwriteMode::Deny,
                    allow_extend_support: false,
                    append_strategy: AppendStrategy::None,
                },
            )
        }

        let cfg = EngineUpdateConfig {
            n_iters: 10,
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::Gibbs),
                StateTransition::StateAlpha,
                StateTransition::RowAssignment(RowAssignAlg::Gibbs),
                StateTransition::ViewAlphas,
                StateTransition::FeaturePriors,
            ],
            ..Default::default()
        };

        let mut engine = EngineBuilder::new(DataSource::Empty).build().unwrap();
        assert_eq!(engine.nrows(), 0);
        assert_eq!(engine.ncols(), 0);

        add_row(&mut engine, "v1", 1.0, 2.0).unwrap();
        add_row(&mut engine, "v2", -1.0, -2.0).unwrap();
        add_row(&mut engine, "v3", 0.0, 0.0).unwrap();
        assert_eq!(engine.nrows(), 3);
        assert_eq!(engine.ncols(), 2);

        engine.update(cfg.clone());

        add_row(&mut engine, "b1", 1.0, 0.5).unwrap();

        assert_eq!(engine.nrows(), 4);
        assert_eq!(engine.ncols(), 2);
        engine.update(cfg.clone());
        assert_eq!(engine.nrows(), 4);

        add_row(&mut engine, "b2", -1.0, 0.1).unwrap();

        assert_eq!(engine.nrows(), 5);
        engine.update(cfg.clone());
        assert_eq!(engine.nrows(), 5);

        add_row(&mut engine, "b3", 0.0, -1.2).unwrap();

        assert_eq!(engine.nrows(), 6);
        engine.update(cfg);
        assert_eq!(engine.nrows(), 6);
    }

    #[test]
    fn insert_empty_row_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![
            Row {
                row_name: "vampire".into(),
                values: vec![Value {
                    col_name: "fast".into(),
                    value: Datum::Categorical(1),
                }],
            },
            Row {
                row_name: "unicorn".into(),
                values: vec![],
            },
        ];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewColumns,
                overwrite: OverwriteMode::Deny,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::EmptyRow(String::from("unicorn"))
        );
    }

    #[test]
    fn insert_empty_single_row_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "unicorn".into(),
            values: vec![],
        }];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewColumns,
                overwrite: OverwriteMode::Deny,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::EmptyRow(String::from("unicorn"))
        );
    }

    #[test]
    #[allow(irrefutable_let_patterns)]
    fn insert_ternary_into_binary_inserts_data() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "pig".into(),
            values: vec![Value {
                col_name: "fierce".into(),
                value: Datum::Categorical(2),
            }],
        }];

        let actions = engine
            .insert_data(
                rows,
                None,
                None,
                WriteMode {
                    insert: InsertMode::DenyNewRowsAndColumns,
                    overwrite: OverwriteMode::Allow,
                    allow_extend_support: true,
                    append_strategy: AppendStrategy::None,
                },
            )
            .unwrap();

        let x = engine
            .datum(animals::Row::Pig.into(), animals::Column::Fierce.into())
            .unwrap();

        assert_eq!(x, Datum::Categorical(2));
        assert!(actions.new_rows().is_none());
        assert!(actions.new_cols().is_none());

        if let Some(suppext) = actions.support_extensions() {
            assert_eq!(suppext.len(), 1);
            if let SupportExtension::Categorical {
                col_ix,
                col_name,
                k_orig,
                k_ext,
            } = &suppext[0]
            {
                assert_eq!(*col_ix, 78);
                assert_eq!(col_name.clone(), String::from("fierce"));
                assert_eq!(*k_orig, 2);
                assert_eq!(*k_ext, 3);
            } else {
                panic!("Wrong kind of support extension");
            }
        } else {
            panic!("Actions does not show support extension");
        }
    }

    #[test]
    fn insert_ternary_into_binary_when_disallowed_errors() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "pig".into(),
            values: vec![Value {
                col_name: "fierce".into(),
                value: Datum::Categorical(2),
            }],
        }];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::Allow,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::ModeForbidsCategoryExtension,
        )
    }

    #[test]
    fn insert_ternary_into_binary_zero_likelihood() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "pig".into(),
            values: vec![Value {
                col_name: "fierce".into(),
                value: Datum::Categorical(2),
            }],
        }];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::Allow,
                allow_extend_support: true,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_ok());

        let surp = engine
            .self_surprisal(
                animals::Row::Pig.into(),
                animals::Column::Fierce.into(),
                None,
            )
            .unwrap()
            .unwrap();

        // new categorical weights are assigned to log(0) by default.
        // Weights are updated when inference is run. This becomes NaN when run
        // through logsumexp.
        assert!(surp.is_nan());
    }

    #[test]
    fn insert_ternary_into_binary_then_run_smoke() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "pig".into(),
            values: vec![Value {
                col_name: "fierce".into(),
                value: Datum::Categorical(2),
            }],
        }];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::Allow,
                allow_extend_support: true,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_ok());
        engine.run(5);
    }

    #[test]
    fn insert_ternary_into_binary_logp_after_run_is_normal() {
        let mut engine = Example::Animals.engine().unwrap();

        let rows = vec![Row {
            row_name: "pig".into(),
            values: vec![Value {
                col_name: "fierce".into(),
                value: Datum::Categorical(2),
            }],
        }];

        let result = engine.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::Allow,
                allow_extend_support: true,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_ok());

        engine.run(2);

        let surp = engine
            .self_surprisal(
                animals::Row::Pig.into(),
                animals::Column::Fierce.into(),
                None,
            )
            .unwrap()
            .unwrap();

        // new categorical weights are assigned to log(0) by default.
        // Weights are updated when inference is run. This becomes NaN when run
        // through logsumexp.
        assert!(surp.is_finite());
        assert!(surp > 0.0);
    }

    macro_rules! update_after_ternary_insert {
        ($test_name: ident, $row_alg: ident, $col_alg: ident) => {
            #[test]
            fn $test_name() {
                use braid::cc::StateTransition;

                let mut engine = Example::Animals.engine().unwrap();

                let rows = vec![Row {
                    row_name: "pig".into(),
                    values: vec![Value {
                        col_name: "fierce".into(),
                        value: Datum::Categorical(2),
                    }],
                }];

                let result = engine.insert_data(
                    rows,
                    None,
                    None,
                    WriteMode {
                        insert: InsertMode::DenyNewRowsAndColumns,
                        overwrite: OverwriteMode::Allow,
                        allow_extend_support: true,
                        append_strategy: AppendStrategy::None,
                    },
                );

                assert!(result.is_ok());
                engine.update(EngineUpdateConfig {
                    n_iters: 2,
                    transitions: vec![
                        StateTransition::StateAlpha,
                        StateTransition::ViewAlphas,
                        StateTransition::ComponentParams,
                        StateTransition::FeaturePriors,
                        StateTransition::RowAssignment(RowAssignAlg::$row_alg),
                        StateTransition::ColumnAssignment(
                            ColAssignAlg::$col_alg,
                        ),
                    ],
                    ..Default::default()
                })
            }
        };
    }

    update_after_ternary_insert!(
        after_ternary_extension_gibbs_gibbs,
        Gibbs,
        Gibbs
    );

    update_after_ternary_insert!(
        after_ternary_extension_sams_gibbs,
        Sams,
        Gibbs
    );

    update_after_ternary_insert!(
        after_ternary_extension_finite_gibbs,
        FiniteCpu,
        Gibbs
    );

    update_after_ternary_insert!(
        after_ternary_extension_slice_gibbs,
        Slice,
        Gibbs
    );

    //
    update_after_ternary_insert!(
        after_ternary_extension_gibbs_finite,
        Gibbs,
        FiniteCpu
    );

    update_after_ternary_insert!(
        after_ternary_extension_sams_finite,
        Sams,
        FiniteCpu
    );

    update_after_ternary_insert!(
        after_ternary_extension_finite_finite,
        FiniteCpu,
        FiniteCpu
    );

    update_after_ternary_insert!(
        after_ternary_extension_slice_finite,
        Slice,
        FiniteCpu
    );

    //
    update_after_ternary_insert!(
        after_ternary_extension_gibbs_slice,
        Gibbs,
        Slice
    );

    update_after_ternary_insert!(
        after_ternary_extension_sams_slice,
        Sams,
        Slice
    );

    update_after_ternary_insert!(
        after_ternary_extension_finite_slice,
        FiniteCpu,
        Slice
    );

    update_after_ternary_insert!(
        after_ternary_extension_slice_slice,
        Slice,
        Slice
    );

    #[test]
    fn insert_extend_categorical_support_with_value_map_column() {
        let mut engine = Example::Satellites.engine().unwrap();

        let rows = vec![Row {
            row_name: "starship enterprise".into(),
            values: vec![Value {
                col_name: "Class_of_Orbit".into(),
                value: Datum::Categorical(2),
            }],
        }];

        let suppl_metadata = {
            let suppl_value_map = btreemap! {
                0 => String::from("Elliptical"),
                1 => String::from("GEO"),
                2 => String::from("MEO"),
                3 => String::from("LEO"),
                4 => String::from("Star Trek"),
            };

            let colmd = ColMetadata {
                name: "Class_of_Orbit".into(),
                notes: None,
                coltype: ColType::Categorical {
                    k: 5,
                    hyper: None,
                    value_map: Some(suppl_value_map),
                    prior: None,
                },
            };

            hashmap! {
                "Class_of_Orbit".into() => colmd
            }
        };

        let result = engine.insert_data(
            rows,
            None,
            Some(suppl_metadata),
            WriteMode {
                insert: InsertMode::DenyNewColumns,
                overwrite: OverwriteMode::Deny,
                allow_extend_support: true,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(result.is_ok());
    }

    fn continuous_md(name: String) -> ColMetadata {
        use braid_stats::prior::ng::NgHyper;

        ColMetadata {
            name,
            coltype: ColType::Continuous {
                hyper: Some(NgHyper::default()),
                prior: None,
            },
            notes: None,
        }
    }

    macro_rules! bad_data_test_existing {
        ($fn_name:ident, $value:expr) => {
            #[test]
            fn $fn_name() {
                let mut engine =
                    EngineBuilder::new(DataSource::Empty).build().unwrap();
                let new_metadata = ColMetadataList::new(vec![
                    continuous_md("one".to_string()),
                    continuous_md("two".to_string()),
                    continuous_md("three".to_string()),
                ])
                .unwrap();

                let rows = vec![
                    Row::from((
                        String::from("1"),
                        vec![
                            (String::from("one"), Datum::Continuous(1.0)),
                            (String::from("two"), Datum::Continuous(2.0)),
                            (String::from("three"), Datum::Continuous(1.0)),
                        ],
                    )),
                    Row::from((
                        String::from("2"),
                        vec![
                            (String::from("one"), Datum::Continuous(1.0)),
                            (String::from("two"), Datum::Continuous(2.0)),
                            (String::from("three"), Datum::Continuous(1.0)),
                        ],
                    )),
                ];

                {
                    let res = engine.insert_data(
                        rows.into(),
                        Some(new_metadata),
                        None,
                        WriteMode::unrestricted(),
                    );
                    assert!(res.is_ok());
                }

                {
                    let rows = vec![Row::from((
                        "3",
                        vec![("one", Datum::Continuous($value))],
                    ))];
                    let err = engine
                        .insert_data(
                            rows.into(),
                            None,
                            None,
                            WriteMode::unrestricted(),
                        )
                        .unwrap_err();
                    if let InsertDataError::NonFiniteContinuousValue {
                        col,
                        value,
                    } = err
                    {
                        assert_eq!(col, String::from("one"));
                        assert!(!value.is_finite());
                    } else {
                        panic!("wrong error");
                    }
                }
            }
        };
    }

    bad_data_test_existing!(
        insert_bad_data_existing_pos_inf,
        std::f64::INFINITY
    );
    bad_data_test_existing!(
        insert_bad_data_existing_neg_inf,
        std::f64::NEG_INFINITY
    );
    bad_data_test_existing!(insert_bad_data_existing_nan, std::f64::NAN);

    macro_rules! bad_data_test_new {
        ($fn_name:ident, $value:expr) => {
            #[test]
            fn $fn_name() {
                let mut engine =
                    EngineBuilder::new(DataSource::Empty).build().unwrap();

                let new_metadata = ColMetadataList::new(vec![
                    continuous_md("one".to_string()),
                    continuous_md("two".to_string()),
                    continuous_md("three".to_string()),
                ])
                .unwrap();

                let rows = vec![
                    Row::from((
                        String::from("1"),
                        vec![
                            (String::from("one"), Datum::Continuous(1.0)),
                            (String::from("two"), Datum::Continuous(2.0)),
                            (String::from("three"), Datum::Continuous(1.0)),
                        ],
                    )),
                    Row::from((
                        String::from("2"),
                        vec![
                            (String::from("one"), Datum::Continuous(1.0)),
                            (String::from("two"), Datum::Continuous(2.0)),
                            (String::from("three"), Datum::Continuous(1.0)),
                        ],
                    )),
                ];

                {
                    let res = engine.insert_data(
                        rows.into(),
                        Some(new_metadata),
                        None,
                        WriteMode::unrestricted(),
                    );
                    assert!(res.is_ok());
                }

                {
                    let rows = vec![Row::from((
                        "3",
                        vec![("fwee", Datum::Continuous($value))],
                    ))];

                    let col_mds = ColMetadataList::new(vec![continuous_md(
                        "fwee".to_string(),
                    )])
                    .unwrap();

                    let err = engine
                        .insert_data(
                            rows.into(),
                            Some(col_mds),
                            None,
                            WriteMode::unrestricted(),
                        )
                        .unwrap_err();

                    if let InsertDataError::NonFiniteContinuousValue {
                        col,
                        value,
                    } = err
                    {
                        assert_eq!(col, String::from("fwee"));
                        assert!(!value.is_finite());
                    } else {
                        panic!("wrong error");
                    }
                }
            }
        };
    }

    bad_data_test_new!(insert_bad_data_new_pos_inf, std::f64::INFINITY);
    bad_data_test_new!(insert_bad_data_new_neg_inf, std::f64::NEG_INFINITY);
    bad_data_test_new!(insert_bad_data_new_nan, std::f64::NAN);

    #[test]
    fn append_single_with_maintain_nrows() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        let fishy = Row::from((
            String::from("fishy"),
            vec![
                (String::from("swims"), Datum::Categorical(1)),
                (String::from("flippers"), Datum::Categorical(1)),
            ],
        ));

        let mode = WriteMode {
            append_strategy: AppendStrategy::Window,
            ..WriteMode::unrestricted()
        };

        engine.insert_data(vec![fishy], None, None, mode).unwrap();
        assert_eq!(engine.nrows(), starting_rows);
    }

    #[test]
    fn append_multiple_with_maintain_nrows() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        let fishy = Row::from((
            String::from("fishy"),
            vec![
                (String::from("swims"), Datum::Categorical(1)),
                (String::from("flippers"), Datum::Categorical(1)),
            ],
        ));

        let rock = Row::from((
            String::from("rock"),
            vec![
                (String::from("swims"), Datum::Categorical(0)),
                (String::from("flippers"), Datum::Categorical(0)),
            ],
        ));

        let mode = WriteMode {
            append_strategy: AppendStrategy::Window,
            ..WriteMode::unrestricted()
        };

        engine
            .insert_data(vec![fishy, rock], None, None, mode)
            .unwrap();
        assert_eq!(engine.nrows(), starting_rows);
    }

    macro_rules! windowed_insert_then_update_smoke {
        ($fn_name:ident, $col_kernel:expr, $row_kernel:expr) => {
            #[test]
            fn $fn_name() {
                let mut engine = Example::Animals.engine().unwrap();
                let starting_rows = engine.nrows();

                let fishy = Row::from((
                    String::from("fishy"),
                    vec![
                        (String::from("swims"), Datum::Categorical(1)),
                        (String::from("flippers"), Datum::Categorical(1)),
                    ],
                ));

                let rock = Row::from((
                    String::from("rock"),
                    vec![
                        (String::from("swims"), Datum::Categorical(0)),
                        (String::from("flippers"), Datum::Categorical(0)),
                    ],
                ));

                let mode = WriteMode {
                    append_strategy: AppendStrategy::Window,
                    ..WriteMode::unrestricted()
                };

                engine
                    .insert_data(vec![fishy, rock], None, None, mode)
                    .unwrap();

                assert_eq!(engine.nrows(), starting_rows);

                let cfg = EngineUpdateConfig {
                    n_iters: 2,
                    transitions: vec![
                        StateTransition::ColumnAssignment($col_kernel),
                        StateTransition::StateAlpha,
                        StateTransition::RowAssignment($row_kernel),
                        StateTransition::ViewAlphas,
                        StateTransition::FeaturePriors,
                    ],
                    ..Default::default()
                };

                engine.update(cfg);

                assert_eq!(engine.nrows(), starting_rows);
            }
        };
    }
    windowed_insert_then_update_smoke!(
        windowed_append_gibbs_gibbs,
        ColAssignAlg::Gibbs,
        RowAssignAlg::Gibbs
    );
    windowed_insert_then_update_smoke!(
        windowed_append_gibbs_sams,
        ColAssignAlg::Gibbs,
        RowAssignAlg::Sams
    );
    windowed_insert_then_update_smoke!(
        windowed_append_gibbs_finite,
        ColAssignAlg::Gibbs,
        RowAssignAlg::FiniteCpu
    );
    windowed_insert_then_update_smoke!(
        windowed_append_gibbs_slice,
        ColAssignAlg::Gibbs,
        RowAssignAlg::Slice
    );

    windowed_insert_then_update_smoke!(
        windowed_append_slice_gibbs,
        ColAssignAlg::Slice,
        RowAssignAlg::Gibbs
    );
    windowed_insert_then_update_smoke!(
        windowed_append_slice_sams,
        ColAssignAlg::Slice,
        RowAssignAlg::Sams
    );
    windowed_insert_then_update_smoke!(
        windowed_append_slice_finite,
        ColAssignAlg::Slice,
        RowAssignAlg::FiniteCpu
    );
    windowed_insert_then_update_smoke!(
        windowed_append_slice_slice,
        ColAssignAlg::Slice,
        RowAssignAlg::Slice
    );

    windowed_insert_then_update_smoke!(
        windowed_append_finite_gibbs,
        ColAssignAlg::FiniteCpu,
        RowAssignAlg::Gibbs
    );
    windowed_insert_then_update_smoke!(
        windowed_append_finite_sams,
        ColAssignAlg::FiniteCpu,
        RowAssignAlg::Sams
    );
    windowed_insert_then_update_smoke!(
        windowed_append_finite_finite,
        ColAssignAlg::FiniteCpu,
        RowAssignAlg::FiniteCpu
    );
    windowed_insert_then_update_smoke!(
        windowed_append_finite_slice,
        ColAssignAlg::FiniteCpu,
        RowAssignAlg::Slice
    );
}

mod del_rows {
    use super::*;
    use braid::HasData;
    use braid::OracleT;

    #[test]
    fn del_first_row() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        let first_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(0, ix).to_u8_opt().unwrap())
            .collect();

        let second_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(1, ix).to_u8_opt().unwrap())
            .collect();

        assert!(first_row.iter().zip(second_row.iter()).any(|(x, y)| x != y));

        engine.del_rows_at(0, 1);

        let new_first_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(0, ix).to_u8_opt().unwrap())
            .collect();

        assert_eq!(engine.nrows(), starting_rows - 1);
        assert!(new_first_row
            .iter()
            .zip(second_row.iter())
            .all(|(x, y)| x == y));
    }

    #[test]
    fn del_first_2_rows() {
        let mut engine = Example::Animals.engine().unwrap();
        let starting_rows = engine.nrows();

        let first_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(0, ix).to_u8_opt().unwrap())
            .collect();

        let third_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(2, ix).to_u8_opt().unwrap())
            .collect();

        assert!(first_row.iter().zip(third_row.iter()).any(|(x, y)| x != y));

        engine.del_rows_at(0, 2);

        let new_first_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(0, ix).to_u8_opt().unwrap())
            .collect();

        assert_eq!(engine.nrows(), starting_rows - 2);
        assert!(new_first_row
            .iter()
            .zip(third_row.iter())
            .all(|(x, y)| x == y));
    }

    #[test]
    fn del_last_row() {
        let mut engine = Example::Animals.engine().unwrap();
        let nrows = engine.nrows();

        let last_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(nrows - 1, ix).to_u8_opt().unwrap())
            .collect();

        let penultimate_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(nrows - 2, ix).to_u8_opt().unwrap())
            .collect();

        assert!(last_row
            .iter()
            .zip(penultimate_row.iter())
            .any(|(x, y)| x != y));

        engine.del_rows_at(nrows - 1, 1);

        let new_last_row: Vec<u8> = (0..engine.ncols())
            .map(|ix| engine.cell(nrows - 2, ix).to_u8_opt().unwrap())
            .collect();

        assert_eq!(engine.nrows(), nrows - 1);
        assert!(new_last_row
            .iter()
            .zip(penultimate_row.iter())
            .all(|(x, y)| x == y));
    }

    #[test]
    fn del_rest_of_rows() {
        let mut engine = Example::Animals.engine().unwrap();
        let nrows = engine.nrows();

        engine.del_rows_at(nrows - 4, 4);

        assert_eq!(engine.nrows(), nrows - 4);
    }

    #[test]
    fn del_last_n_rows_deletes_up_to_last_row() {
        let mut engine = Example::Animals.engine().unwrap();
        let nrows = engine.nrows();

        engine.del_rows_at(nrows - 5, 10);

        assert_eq!(engine.nrows(), nrows - 5);
    }
}

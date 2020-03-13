use std::convert::Into;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use braid::data::DataSource;
use braid::examples::Example;
use braid::{Engine, EngineBuilder};
use braid_codebook::Codebook;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

const ANIMALS_DATA: &str = "resources/datasets/animals/data.csv";
const ANIMALS_CODEBOOK: &str = "resources/datasets/animals/codebook.yaml";

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
        let mut file = File::open(ANIMALS_CODEBOOK).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        serde_yaml::from_slice(data.as_bytes()).unwrap()
    };
    let rng = Xoshiro256Plus::from_entropy();
    match Engine::new(0, codebook, DataSource::Csv(ANIMALS_DATA.into()), 0, rng)
    {
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

// NOTE: These tests make sure that values have been updated, that the desired
// rows and columns have been added, and that bad inputs return the correct
// errors. They do not make sure the State metadata (assignment and sufficient
// statistics) have been updated properly. Those tests occur in State.
mod insert_data {
    use super::*;
    use braid::error::InsertDataError;
    use braid::{InsertMode, InsertOverwrite, OracleT, Row, Value};
    use braid_codebook::{ColMetadata, ColMetadataList, ColType, SpecType};
    use braid_stats::Datum;

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

        let result = engine.insert_data(
            rows,
            None,
            InsertMode::DenyNewColumns(InsertOverwrite::Deny),
        );

        assert!(result.is_ok());
        assert_eq!(engine.nrows(), starting_rows + 1);
        assert_eq!(engine.ncols(), starting_cols);

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

            let result = engine.insert_data(
                rows,
                None,
                InsertMode::DenyNewColumns(InsertOverwrite::Deny),
            );

            assert!(result.is_ok());
            assert_eq!(engine.nrows(), starting_rows + 1);
        }

        {
            let rows = vec![Row {
                row_name: "yoshi".into(),
                values: vec![Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(0),
                }],
            }];

            let result = engine.insert_data(
                rows,
                None,
                InsertMode::DenyNewColumns(InsertOverwrite::Deny),
            );

            assert!(result.is_ok());
            assert_eq!(engine.nrows(), starting_rows + 2);
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

            let result = engine.insert_data(
                rows,
                None,
                InsertMode::DenyNewColumns(InsertOverwrite::Deny),
            );

            assert!(result.is_ok());
            assert_eq!(engine.nrows(), starting_rows + 1);
        }

        {
            let rows = vec![Row {
                row_name: "pegasus".into(),
                values: vec![Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(0),
                }],
            }];

            let result = engine.insert_data(
                rows,
                None,
                InsertMode::DenyNewRowsAndColumns(InsertOverwrite::MissingOnly),
            );

            assert!(result.is_ok());
            assert_eq!(engine.nrows(), starting_rows + 1);
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

        let result = engine.insert_data(
            rows,
            None,
            InsertMode::DenyNewRowsAndColumns(InsertOverwrite::Allow),
        );

        assert!(result.is_ok());
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

        let result = engine.insert_data(
            rows,
            None,
            InsertMode::DenyNewRowsAndColumns(InsertOverwrite::Allow),
        );

        assert!(result.is_ok());
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
            spec_type: SpecType::Other,
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        assert_eq!(engine.ncols(), 85);

        let result = engine.insert_data(
            rows,
            Some(col_metadata),
            InsertMode::DenyNewRows(InsertOverwrite::Deny),
        );

        assert!(result.is_ok());
        assert_eq!(engine.nrows(), starting_rows);
        assert_eq!(engine.ncols(), 86);

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
            spec_type: SpecType::Other,
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        assert_eq!(engine.ncols(), 85);

        let result = engine.insert_data(
            rows,
            Some(col_metadata),
            InsertMode::Unrestricted(InsertOverwrite::Deny),
        );

        assert!(result.is_ok());
        assert_eq!(engine.nrows(), 51);
        assert_eq!(engine.ncols(), 86);

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
            InsertMode::DenyNewRowsAndColumns(InsertOverwrite::Deny),
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
            InsertMode::DenyNewRowsAndColumns(InsertOverwrite::MissingOnly),
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
            spec_type: SpecType::Other,
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        let result = engine.insert_data(
            rows,
            Some(col_metadata),
            InsertMode::DenyNewColumns(InsertOverwrite::Deny),
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
            spec_type: SpecType::Other,
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        let result = engine.insert_data(
            rows,
            Some(col_metadata),
            InsertMode::DenyNewRows(InsertOverwrite::Deny),
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
            InsertMode::DenyNewRows(InsertOverwrite::Allow),
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
            spec_type: SpecType::Other,
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        engine
            .insert_data(
                rows,
                Some(col_metadata),
                InsertMode::Unrestricted(InsertOverwrite::Deny),
            )
            .unwrap();

        engine.run(5)
    }

    #[test]
    fn insert_into_empty() {
        use braid_stats::prior::NigHyper;
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
            hyper: Some(NigHyper {
                pr_m: Gaussian::new_unchecked(0.0, 1.0),
                pr_r: Gamma::new_unchecked(2.0, 1.0),
                pr_s: Gamma::new_unchecked(1.0, 1.0),
                pr_v: Gamma::new_unchecked(2.0, 1.0),
            }),
        };

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "score".to_string(),
            spec_type: SpecType::Phenotype,
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

        engine
            .insert_data(
                vec![row],
                Some(col_metadata),
                InsertMode::Unrestricted(InsertOverwrite::Allow),
            )
            .expect("Failed to insert data");

        assert_eq!(engine.nrows(), 1);
        assert_eq!(engine.ncols(), 1);
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
                InsertMode::DenyNewColumns(InsertOverwrite::Deny),
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
            spec_type: SpecType::Other,
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        engine
            .insert_data(
                new_col,
                Some(col_metadata),
                InsertMode::DenyNewRows(InsertOverwrite::Deny),
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
                spec_type: SpecType::Other,
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    value_map: None,
                },
                notes: None,
            },
            ColMetadata {
                name: "fierce".into(),
                spec_type: SpecType::Other,
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
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
                InsertMode::Unrestricted(InsertOverwrite::Deny),
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
}

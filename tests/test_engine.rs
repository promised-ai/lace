use std::convert::Into;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use braid::data::DataSource;
use braid::error;
use braid::{Engine, EngineBuilder, RowAlignmentStrategy};
use braid_codebook::Codebook;
use braid_stats::Datum;
use indoc::indoc;
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
        Err(braid::error::NewEngineError::ZeroStatesRequestedError) => (),
        Err(_) => panic!("wrong error"),
        Ok(_) => panic!("Failed to catch zero states error"),
    }
}

#[test]
fn append_row() {
    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nstates(), 2);
    println!("{:?}", engine.state_ids);
    assert_eq!(engine.states[0].nrows(), 3);

    let new_rows =
        DataSource::Csv("resources/test/small/small-one-more.csv".into());
    engine.append_rows(new_rows).unwrap();

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states[0].nrows(), 4);
    assert_eq!(engine.codebook.row_names[3], String::from("D"));

    for state in engine.states.iter() {
        let x_0 = state.datum(3, 0).to_u8_opt().unwrap();
        let x_1 = state.datum(3, 1).to_u8_opt().unwrap();
        let x_2 = state.datum(3, 2).to_u8_opt().unwrap();

        assert_eq!(x_0, 1);
        assert_eq!(x_1, 0);
        assert_eq!(x_2, 1);
    }
}

#[test]
fn append_rows() {
    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states[0].nrows(), 3);

    let new_rows =
        DataSource::Csv("resources/test/small/small-two-more.csv".into());

    engine.append_rows(new_rows).unwrap();

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states[0].nrows(), 5);

    let row_names = engine.codebook.row_names;

    assert_eq!(row_names[3], String::from("D"));
    assert_eq!(row_names[4], String::from("E"));

    for state in engine.states.iter() {
        let x_30 = state.datum(3, 0).to_u8_opt().unwrap();
        let x_31 = state.datum(3, 1).to_u8_opt().unwrap();
        let x_32 = state.datum(3, 2).to_u8_opt().unwrap();

        assert_eq!(x_30, 1);
        assert_eq!(x_31, 0);
        assert_eq!(x_32, 1);

        let x_40 = state.datum(4, 0).to_u8_opt().unwrap();
        let x_41 = state.datum(4, 1);
        let x_42 = state.datum(4, 2);

        assert_eq!(x_40, 0);
        assert_eq!(x_41, Datum::Missing);
        assert_eq!(x_42, Datum::Missing);
    }
}

#[test]
fn append_rows_with_nonexisting_file_causes_io_error() {
    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states[0].nrows(), 3);

    let new_rows =
        DataSource::Csv("resources/test/small/file-not-found.csv".into());

    match engine.append_rows(new_rows) {
        Err(braid::error::AppendRowsError::IoError) => (),
        Err(_) => panic!("wrong error"),
        Ok(_) => panic!("Somehow succeeded with no data"),
    }
}

#[test]
fn append_rows_with_postgres_causes_unsupported_type_error() {
    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states[0].nrows(), 3);

    let new_rows = DataSource::Postgres("shouldnt_matter.pg".into());

    match engine.append_rows(new_rows) {
        Err(braid::error::AppendRowsError::UnsupportedDataSourceError) => (),
        Err(_) => panic!("wrong error"),
        Ok(_) => panic!("Somehow succeeded with no data"),
    }
}

#[test]
fn append_rows_with_missing_columns_csv_causes_row_lenth_error() {
    use braid::data::CsvParseError;
    use braid::error::DataParseError;
    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states[0].nrows(), 3);

    let mut file = tempfile::NamedTempFile::new().unwrap();
    let new_rows = {
        let raw = "\
            id,two,three
            D,0,1
            E,1,0\
        ";
        file.write(raw.as_bytes()).unwrap();
        DataSource::Csv(file.path().into())
    };

    match engine.append_rows(new_rows) {
        Err(error::AppendRowsError::DataParseError(
            DataParseError::CsvParseError(
                CsvParseError::MissingCsvColumnsError,
            ),
        )) => (),
        Err(err) => panic!("wrong error: {:?}", err),
        Ok(_) => panic!("Somehow succeeded with no data"),
    }
}

fn write_to_tempfile(s: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write(s.as_bytes()).unwrap();
    file
}

#[test]
fn append_features_should_add_features() {
    let new_cols = "\
        id,four,five
        A,0,1
        B,0,1
        C,0,1\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "five"
            coltype:
              Categorical:
                k: 2
        row_names:
          - 0
          - 1
          - 2
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::Ignore,
    );

    assert!(result.is_ok());
    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 5);
}

#[test]
fn append_features_with_correct_row_names_should_add_features_if_check() {
    let new_cols = "\
        id,four,five
        A,0,1
        B,0,1
        C,0,1\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "five"
            coltype:
              Categorical:
                k: 2
        row_names:
          - A
          - B
          - C
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::Ignore,
    );

    assert!(result.is_ok());
    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 5);
}

#[test]
fn append_features_with_wrong_number_of_rows_should_error() {
    use braid::error::AppendFeaturesError;
    let new_cols = "\
        id,four,five
        A,0,1
        B,0,1\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "five"
            coltype:
              Categorical:
                k: 2
        row_names:
          - 0
          - 1
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::Ignore,
    );

    assert_eq!(result, Err(AppendFeaturesError::ColumnLengthError));
}

#[test]
fn append_features_with_duplicate_column_should_error() {
    use braid::error::AppendFeaturesError;
    // The column "two" appears in small.csv
    let new_cols = "\
        id,four,two
        A,0,1
        B,0,1
        C,1,0\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "two"
            coltype:
              Categorical:
                k: 2
        row_names:
          - 0
          - 1
          - 2
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::Ignore,
    );

    assert_eq!(
        result,
        Err(AppendFeaturesError::ColumnAlreadyExistsError(String::from(
            "two"
        )))
    );
}

#[test]
fn append_features_with_mismatched_col_names_in_files_should_error() {
    use braid::error::AppendFeaturesError;

    // The column "five" appears in the csv, but does not appear in the codebook
    let new_cols = "\
        id,four,five
        A,0,1
        B,0,1\
        C,1,0\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "fiver"
            coltype:
              Categorical:
                k: 2
        row_names:
          - 0
          - 1
          - 2
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::Ignore,
    );

    assert_eq!(
        result,
        Err(AppendFeaturesError::CodebookDataColumnNameMismatchError)
    );
}

#[test]
fn append_features_with_bad_source_should_error() {
    use braid::error::AppendFeaturesError;

    // The column "five" appears in the csv, but does not appear in the codebook
    let data_src = DataSource::Csv("doesnt-exist.csv".into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "fiver"
            coltype:
              Categorical:
                k: 2
        row_names:
          - 0
          - 1
          - 2
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::Ignore,
    );

    assert_eq!(result, Err(AppendFeaturesError::IoError));
}

#[test]
fn append_features_with_wrong_num_rownames_errs_if_name_check() {
    use braid::error::AppendFeaturesError;
    let new_cols = "\
        id,four,five
        A,0,1
        B,0,1
        C,0,1\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "five"
            coltype:
              Categorical:
                k: 2
        row_names:
          - 0
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::CheckNames,
    );

    assert_eq!(result, Err(AppendFeaturesError::RowNameMismatchError));
}

#[test]
fn append_features_with_wrong_rownames_errs_if_name_check() {
    use braid::error::AppendFeaturesError;
    let new_cols = "\
        id,four,five
        A,0,1
        B,0,1
        C,0,1\
    ";
    let file = write_to_tempfile(new_cols);
    let data_src = DataSource::Csv(file.path().into());
    let codebook_str = indoc!(
        r#"
        ---
        table_name: test
        col_metadata:
          - name: "four"
            coltype:
              Categorical:
                k: 2
          - name: "five"
            coltype:
              Categorical:
                k: 2
        row_names:
          - A
          - B
          - F
        "#
    );

    let partial_codebook: Codebook =
        serde_yaml::from_str(codebook_str).unwrap();

    let mut engine = engine_from_csv("resources/test/small/small.csv");

    assert_eq!(engine.nrows(), 3);
    assert_eq!(engine.ncols(), 3);

    let result = engine.append_features(
        partial_codebook,
        data_src,
        RowAlignmentStrategy::CheckNames,
    );

    assert_eq!(result, Err(AppendFeaturesError::RowNameMismatchError));
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
    use braid::examples::Example;
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
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::ModeForbidsOverwriteError
        );
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
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::ModeForbidsOverwriteError
        );
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
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::ModeForbidsNewColumnsError
        );
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
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::ModeForbidsNewRowsError
        );
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
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::ModeForbidsNewRowsError
        );
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

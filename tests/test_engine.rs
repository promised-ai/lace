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
    println!("{:?}", engine.states.keys());
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

    let new_rows =
        DataSource::Csv("resources/test/small/small-one-more.csv".into());
    engine.append_rows(new_rows).unwrap();

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 4);
    assert_eq!(engine.codebook.row_names.unwrap()[3], String::from("D"));

    for state in engine.states.values() {
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
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

    let new_rows =
        DataSource::Csv("resources/test/small/small-two-more.csv".into());

    engine.append_rows(new_rows).unwrap();

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 5);

    let row_names = engine.codebook.row_names.unwrap();

    assert_eq!(row_names[3], String::from("D"));
    assert_eq!(row_names[4], String::from("E"));

    for state in engine.states.values() {
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
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

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
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

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
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

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
fn append_features_with_no_rownames_errs_if_name_check() {
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

    assert_eq!(result, Err(AppendFeaturesError::NoRowNamesInChildError));
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

        for (_, state) in &engine.states {
            assert_eq!(state.diagnostics.loglike.len(), 100);
            assert_eq!(state.diagnostics.nviews.len(), 100);
            assert_eq!(state.diagnostics.state_alpha.len(), 100);
        }

        engine.save_to(dir.as_ref()).save().unwrap();
    }

    {
        let mut engine = Engine::load(dir.as_ref()).unwrap();

        for (_, state) in &engine.states {
            assert_eq!(state.diagnostics.loglike.len(), 100);
            assert_eq!(state.diagnostics.nviews.len(), 100);
            assert_eq!(state.diagnostics.state_alpha.len(), 100);
        }

        engine.run(10);

        for (_, state) in engine.states {
            assert_eq!(state.diagnostics.loglike.len(), 110);
            assert_eq!(state.diagnostics.nviews.len(), 110);
            assert_eq!(state.diagnostics.state_alpha.len(), 110);
        }
    }
}

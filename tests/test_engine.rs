use std::convert::Into;
use std::path::PathBuf;

use braid::data::DataSource;
use braid::{Engine, EngineBuilder};
use braid_stats::Datum;

// TODO: Don't use tiny test files, generate them in code from raw strings and
// tempfiles.
fn engine_from_csv<P: Into<PathBuf>>(path: P) -> Engine {
    EngineBuilder::new(DataSource::Csv(path.into()))
        .with_nstates(2)
        .build()
        .unwrap()
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

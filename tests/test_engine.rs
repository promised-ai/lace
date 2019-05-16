extern crate braid;
extern crate braid_codebook;
extern crate csv;

use braid::data::DataSource;
use braid::Datum;
use braid::{Engine, EngineBuilder};

// TODO: Don't use tiny test files, generate them in code from raw strings and
// tempfiles.
fn engine_from_csv(path: String) -> Engine {
    EngineBuilder::new(DataSource::Csv(path))
        .with_nstates(2)
        .build()
        .unwrap()
}

#[test]
fn append_row() {
    let mut engine = engine_from_csv("resources/test/small.csv".into());

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

    let new_rows = DataSource::Csv("resources/test/small-one-more.csv".into());
    engine.append_rows(new_rows);

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 4);
    assert_eq!(engine.codebook.row_names.unwrap()[3], String::from("D"));

    for state in engine.states.values() {
        let x_0 = state.get_datum(3, 0).as_u8().unwrap();
        let x_1 = state.get_datum(3, 1).as_u8().unwrap();
        let x_2 = state.get_datum(3, 2).as_u8().unwrap();

        assert_eq!(x_0, 1);
        assert_eq!(x_1, 0);
        assert_eq!(x_2, 1);
    }
}

#[test]
fn append_rows() {
    let mut engine = engine_from_csv("resources/test/small.csv".into());

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 3);

    let new_rows = DataSource::Csv("resources/test/small-two-more.csv".into());
    engine.append_rows(new_rows);

    assert_eq!(engine.nstates(), 2);
    assert_eq!(engine.states.get(&0).unwrap().nrows(), 5);

    let row_names = engine.codebook.row_names.unwrap();

    assert_eq!(row_names[3], String::from("D"));
    assert_eq!(row_names[4], String::from("E"));

    for state in engine.states.values() {
        let x_30 = state.get_datum(3, 0).as_u8().unwrap();
        let x_31 = state.get_datum(3, 1).as_u8().unwrap();
        let x_32 = state.get_datum(3, 2).as_u8().unwrap();

        assert_eq!(x_30, 1);
        assert_eq!(x_31, 0);
        assert_eq!(x_32, 1);

        let x_40 = state.get_datum(4, 0).as_u8().unwrap();
        let x_41 = state.get_datum(4, 1);
        let x_42 = state.get_datum(4, 2);

        assert_eq!(x_40, 0);
        assert_eq!(x_41, Datum::Missing);
        assert_eq!(x_42, Datum::Missing);
    }
}

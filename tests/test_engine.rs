extern crate braid;
extern crate braid_codebook;
extern crate csv;

use braid::data::DataSource;
use braid::{Engine, EngineBuilder};

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
}

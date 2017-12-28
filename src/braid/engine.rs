extern crate serde_json;
extern crate serde_yaml;
extern crate rand;
extern crate rusqlite;

use std::path::Path;
use std::fs::File;
use std::io::{Write, Read};

use self::rusqlite::Connection;
use self::rand::Rng;

use cc::{State, Codebook};
use data::{SerializedType, DataSource};
use data::sqlite;



// TODO: templatize rng type
pub struct Engine<R: Rng> {
    rng: R,
    states: Vec<State>,
}


/// The object on which the server acts
impl<R: Rng> Engine<R> {
    pub fn new(nstates: usize, codebook: Codebook, src_path: &Path,
               data_source: DataSource, mut rng: R) -> Self
    {
        let state_alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
        let col_models = match data_source {
            DataSource::Sqlite => {
                // FIXME: Open read-only w/ flags
                let conn = Connection::open(src_path).unwrap();
                sqlite::read_cols(&conn, &codebook)
            },
            DataSource::Postgres => unimplemented!(),
            DataSource::Csv => unimplemented!(),
        };

        let states = (0..nstates).map(|_| {
            let features = col_models.clone();
            State::from_prior(features, state_alpha, &mut rng)
        }).collect();
        Engine { rng: rng, states: states }
    }

    pub fn load(path: &Path, file_type: SerializedType, rng: R) -> Self {
        let mut file = File::open(&path).unwrap();
        let mut ser = String::new();
        file.read_to_string(&mut ser).unwrap();

        let states = match file_type {
            SerializedType::Json => serde_json::from_str(&ser.as_str()).unwrap(),
            SerializedType::Yaml => serde_yaml::from_str(&ser.as_str()).unwrap(),
            SerializedType::MessagePack => unimplemented!(),
        };
        
        Engine { rng: rng, states: states }
    }

    pub fn save(&self, path: &Path, file_type: SerializedType) {
        let states = &self.states;
        let ser = match file_type {
            SerializedType::Json => serde_json::to_string(&states).unwrap(),
            SerializedType::Yaml => serde_yaml::to_string(&states).unwrap(),
            SerializedType::MessagePack => unimplemented!(),
        };
        let mut file = File::create(path).unwrap();
        let _nbytes = file.write(ser.as_bytes()).unwrap();
    }

    pub fn from_sqlite(db_path: &Path, codebook: Codebook, nstates: usize,
                   mut rng: R) -> Self
    {
        let alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
        let conn = Connection::open(&db_path).unwrap();
        let ftrs = sqlite::read_cols(&conn, &codebook);

        let states: Vec<State> = (0..nstates).map(|_| {
            State::from_prior(ftrs.clone(), alpha, &mut rng)
        }).collect();
        Engine { states: states, rng: rng }
    }

    pub fn from_postegres(_path: &Path) -> Self {
        unimplemented!();
    }

    // TODO: Choose row assign and col assign algorithms
    // TODO: Checkpoint for diagnostic collection
    // TODO: Savepoint for intermediate serialization
    // TODO: Run for time.
    pub fn run(&mut self, n_iter: usize, _checkpoint: usize) {
        let mut rng = &mut self.rng;
        self.states
            .iter_mut()
            .for_each(|state| state.update(n_iter, &mut rng));
    }
}

extern crate serde_json;
extern crate serde_yaml;
extern crate rmp_serde;
extern crate rand;
extern crate rusqlite;
extern crate csv;

use std::path::Path;
use std::fs::File;
use std::io::{Write, Read};

use rayon::prelude::*;
use self::rusqlite::Connection;
use self::csv::ReaderBuilder;

use cc::{State, Codebook};
use data::{SerializedType, DataSource};
use data::csv as braid_csv;
use data::sqlite;



// TODO: templatize rng type
#[allow(dead_code)]  // rng is dead
pub struct Engine {
    states: Vec<State>,
}


/// The object on which the server acts
impl Engine {
    pub fn new(nstates: usize, codebook: Codebook, src_path: &Path,
               data_source: DataSource) -> Self
    {
        let state_alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
        let col_models = match data_source {
            DataSource::Sqlite => {
                // FIXME: Open read-only w/ flags
                let conn = Connection::open(src_path).unwrap();
                sqlite::read_cols(&conn, &codebook)
            },
            DataSource::Csv => {
                let mut reader = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(&src_path)
                    .unwrap();
                braid_csv::read_cols(reader, &codebook)
            }
            DataSource::Postgres => unimplemented!(),
        };

        let states = (0..nstates).map(|_| {
            let mut rng = rand::thread_rng();
            let features = col_models.clone();
            State::from_prior(features, state_alpha, &mut rng)
        }).collect();
        Engine { states: states }
    }

    pub fn load(path: &Path, file_type: SerializedType) -> Self {
        let mut file = File::open(&path).unwrap();
        let mut ser = String::new();

        let states = match file_type {
            SerializedType::Json => {
                file.read_to_string(&mut ser).unwrap();
                serde_json::from_str(&ser.as_str()).unwrap()
            },
            SerializedType::Yaml => {
                file.read_to_string(&mut ser).unwrap();
                serde_yaml::from_str(&ser.as_str()).unwrap()
            },
            SerializedType::MessagePack => {
                let mut buf = Vec::new();
                let _res = file.read_to_end(&mut buf);
                rmp_serde::from_slice(&buf.as_slice()).unwrap()
            },
        };
        
        Engine { states: states }
    }

    pub fn save(&self, path: &Path, file_type: SerializedType) {
        let states = &self.states;
        let ser = match file_type {
            SerializedType::Json => {
                serde_json::to_string(&states).unwrap().into_bytes()
            }
            SerializedType::Yaml => {
                serde_yaml::to_string(&states).unwrap().into_bytes()
            },
            SerializedType::MessagePack => {
                rmp_serde::to_vec(&states).unwrap()
            },
        };

        let mut file = File::create(path).unwrap();
        let _nbytes = file.write(&ser).unwrap();
    }

    pub fn from_sqlite(db_path: &Path, codebook: Codebook, nstates: usize) -> Self
    {
        let alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
        let conn = Connection::open(&db_path).unwrap();
        let ftrs = sqlite::read_cols(&conn, &codebook);

        let states: Vec<State> = (0..nstates).map(|_| {
            let mut rng = rand::thread_rng();
            State::from_prior(ftrs.clone(), alpha, &mut rng)
        }).collect();
        Engine { states: states }
    }

    pub fn from_postegres(_path: &Path) -> Self {
        unimplemented!();
    }

    // TODO: Choose row assign and col assign algorithms
    // TODO: Checkpoint for diagnostic collection
    // TODO: Savepoint for intermediate serialization
    // TODO: Run for time.
    pub fn run(&mut self, n_iter: usize, _checkpoint: usize) {
        self.states
            .par_iter_mut()
            .for_each(|state| {
                let mut rng = rand::thread_rng();
                state.update(n_iter, &mut rng);
            });
    }
}

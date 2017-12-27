extern crate rand;
extern crate rusqlite;

use std::path::Path;

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

    pub fn load(path: &Path) -> Self {
        unimplemented!();
    }

    pub fn save(&self, path: &Path, fileType: SerializedType) -> Self {
        unimplemented!();
    }

    fn from_sqlite(path: &Path) -> Self {
        unimplemented!();
    }

    fn from_postegres(path: &Path) -> Self {
        unimplemented!();
    }

    // TODO: Choose row assign and col assign algorithms
    // TODO: Checkpoint for diagnostic collection
    // TODO: Savepoint for intermediate serialization
    // TODO: Run for time.
    pub fn run(&mut self, n_iter: usize) {
        let mut rng = &mut self.rng;
        self.states
            .iter_mut()
            .for_each(|state| state.update(n_iter, &mut rng));
    }
}

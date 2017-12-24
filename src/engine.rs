extern crate rand;

use std::path::Path;

use self::rand::Rng;

use cc::State;
use data::FileType;
use data::sqlite;


// TODO: templatize rng type
pub struct Engine<R: Rng> {
    rng: R,
    states: Vec<State>,
}



/// The object on which the server acts
impl<R: Rng> Engine<R> {
    pub fn load(path: &Path) -> Self {
        unimplemented!();
    }

    pub fn save(&self, path: &Path, fileType: FileType) -> Self {
        unimplemented!();
    }

    pub fn from_sqlite(path: &Path) -> Self {
        unimplemented!();
    }

    pub fn from_postegres(path: &Path) -> Self {
        unimplemented!();
    }

    pub fn run(&mut self) {
        unimplemented!();
    }
}

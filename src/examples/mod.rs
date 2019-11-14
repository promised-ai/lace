pub mod animals;

use crate::data::DataSource;
use crate::{Engine, EngineBuilder, Oracle};
use braid_codebook::codebook::Codebook;
use std::fs::create_dir_all;
use std::io::{self, Read};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum IndexError {
    RowIndexError(usize),
    ColumnIndexError(usize),
}

/// Stores the location of the example's data and codebook
#[derive(Clone)]
struct ExamplePaths {
    data: PathBuf,
    codebook: PathBuf,
    braid: PathBuf,
}

/// Some simple examples for playing with analyses
///
/// # Examples
///
/// ```
/// # use braid::examples::Example;
/// use braid::examples::animals::Row;
///
/// let oracle = Example::Animals.oracle().unwrap();
///
/// let sim_wolf = oracle.rowsim(Row::Chihuahua.into(), Row::Wolf.into(), None)
///     .unwrap();
/// let sim_rat = oracle.rowsim(Row::Chihuahua.into(), Row::Rat.into(), None)
///     .unwrap();
///
/// assert!(sim_wolf < sim_rat);
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Example {
    /// A dataset with animals and their features
    Animals,
}

impl Example {
    fn paths(self) -> io::Result<ExamplePaths> {
        let base_dir = braid_data_dir().map(|dir| dir.join(self.to_str()))?;
        Ok(ExamplePaths {
            data: base_dir.join("data.csv"),
            codebook: base_dir.join("codebook.yaml"),
            braid: base_dir.join("braid"),
        })
    }

    pub fn regen_metadata(self) -> io::Result<()> {
        let paths = self.paths()?;
        let codebook: Codebook = {
            let mut file = std::fs::File::open(&paths.codebook)?;
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_yaml::from_str(&ser.as_str()).map_err(|_| {
                let err_kind = io::ErrorKind::InvalidData;
                io::Error::new(err_kind, "Could not parse codebook")
            })?
        };

        let mut engine: Engine =
            EngineBuilder::new(DataSource::Csv(paths.data))
                .with_codebook(codebook)
                .with_nstates(8)
                .with_seed(1776)
                .build()
                .map_err(|_| {
                    let err_kind = io::ErrorKind::Other;
                    io::Error::new(err_kind, "Failed to create Engine")
                })?;

        engine.run(500);
        engine.save_to(&paths.braid.as_path()).save()?;
        Ok(())
    }

    /// Get an oracle build for the example. If this is the first time using
    /// the example, a new analysis will run. Be patient.
    pub fn oracle(self) -> io::Result<Oracle> {
        let paths = self.paths()?;
        if paths.braid.exists() {
            Oracle::load(paths.braid.as_path())
        } else {
            self.regen_metadata()?;
            Oracle::load(paths.braid.as_path())
        }
    }

    fn to_str(&self) -> &str {
        match self {
            Example::Animals => "animals",
        }
    }
}

/// Creates and returns a data directory
fn braid_data_dir() -> io::Result<PathBuf> {
    let data_dir: PathBuf = dirs::data_dir()
        .map(|dir| dir.join("braid").join("examples"))
        .ok_or({
            let err_kind = io::ErrorKind::NotFound;
            io::Error::new(err_kind, "could not find user data directory")
        })?;

    create_dir_all(&data_dir).map(|_| data_dir)
}

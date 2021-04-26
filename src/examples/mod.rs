pub mod animals;
pub mod satellites;

use crate::data::DataSource;
use crate::{Engine, EngineBuilder, Oracle};
use braid_codebook::Codebook;
use std::fs::create_dir_all;
use std::io::{self, Read};
use std::path::PathBuf;
use thiserror::Error;

const DEFAULT_N_ITERS: usize = 1_000;
const DEFAULT_TIMEOUT: Option<u64> = Some(120);

#[derive(Clone, Debug, Error)]
pub enum IndexConversionError {
    /// The row index is too high
    #[error("cannot convert index {row_ix} into a row for a dataset with {nrows} rows")]
    RowIndexOutOfBounds { row_ix: usize, nrows: usize },
    /// The column index is too high
    #[error("cannot convert index {col_ix} into a column for dataset with {ncols} columns")]
    ColumnIndexOutOfBounds { col_ix: usize, ncols: usize },
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
/// use braid::OracleT;
/// use braid::examples::animals::Row;
///
/// let oracle = Example::Animals.oracle().unwrap();
///
/// let sim_wolf = oracle.rowsim(
///     Row::Chihuahua.into(),
///     Row::Wolf.into(),
///     None,
///     false
/// ).unwrap();
///
/// let sim_rat = oracle.rowsim(
///     Row::Chihuahua.into(),
///     Row::Rat.into(),
///     None,
///     false
/// ).unwrap();
///
/// assert!(sim_wolf < sim_rat);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Example {
    /// A dataset with animals and their features
    Animals,
    /// A dataset of Earth-orbiting satellites with information about their
    /// user, purpose, and orbital characteristics
    Satellites,
}

impl std::str::FromStr for Example {
    type Err = String;

    fn from_str(s: &str) -> Result<Example, Self::Err> {
        match s {
            "animals" => Ok(Self::Animals),
            "satellites" => Ok(Self::Satellites),
            _ => Err(format!("cannot parse '{}' as Example", s)),
        }
    }
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

    pub fn regen_metadata(
        self,
        n_iters: usize,
        timeout: Option<u64>,
    ) -> io::Result<()> {
        use crate::cc::config::EngineUpdateConfig;
        let paths = self.paths()?;
        let codebook: Codebook = {
            let mut file = std::fs::File::open(&paths.codebook)?;
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_yaml::from_str(&ser.as_str()).map_err(|err| {
                eprint!("{:?}", err);
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

        let config = EngineUpdateConfig {
            n_iters,
            timeout,
            ..Default::default()
        };

        engine.update(config);
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
            self.regen_metadata(DEFAULT_N_ITERS, DEFAULT_TIMEOUT)?;
            Oracle::load(paths.braid.as_path())
        }
    }

    /// Get an engine build for the example. If this is the first time using
    /// the example, a new analysis will run. Be patient.
    pub fn engine(self) -> io::Result<Engine> {
        let paths = self.paths()?;
        if paths.braid.exists() {
            Engine::load(paths.braid.as_path())
        } else {
            self.regen_metadata(DEFAULT_N_ITERS, DEFAULT_TIMEOUT)?;
            Engine::load(paths.braid.as_path())
        }
    }

    fn to_str(&self) -> &str {
        match self {
            Example::Animals => "animals",
            Example::Satellites => "satellites",
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
pub mod animals;
pub mod satellites;

use crate::data::DataSource;
use crate::update_handler::Timeout;
use crate::{Engine, Oracle};
use lace_codebook::Codebook;
use lace_metadata::Error;
use std::fs::create_dir_all;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

const DEFAULT_N_ITERS: usize = 5_000;
const DEFAULT_TIMEOUT: Option<u64> = None;

#[derive(Clone, Debug, Error)]
pub enum IndexConversionError {
    /// The row index is too high
    #[error("cannot convert index {row_ix} into a row for a dataset with {n_rows} rows")]
    RowIndexOutOfBounds { row_ix: usize, n_rows: usize },
    /// The column index is too high
    #[error("cannot convert index {col_ix} into a column for dataset with {n_cols} columns")]
    ColumnIndexOutOfBounds { col_ix: usize, n_cols: usize },
}

/// Stores the location of the example's data and codebook
#[derive(Clone)]
struct ExamplePaths {
    data: PathBuf,
    codebook: PathBuf,
    lace: PathBuf,
}

/// Some simple examples for playing with analyses
///
/// # Examples
///
/// ```
/// # use lace::examples::Example;
/// use lace::OracleT;
/// use lace::RowSimilarityVariant;
/// use lace::examples::animals::Row;
///
/// let oracle = Example::Animals.oracle().unwrap();
///
/// let wrt: Option<&[usize]> = None;
/// let sim_wolf = oracle.rowsim(
///     Row::Chihuahua.ix(),
///     Row::Wolf.ix(),
///     wrt,
///     RowSimilarityVariant::ViewWeighted,
/// ).unwrap();
///
/// let sim_rat = oracle.rowsim(
///     Row::Chihuahua.ix(),
///     Row::Rat.ix(),
///     wrt,
///     RowSimilarityVariant::ViewWeighted,
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "animals" => Ok(Self::Animals),
            "satellites" => Ok(Self::Satellites),
            _ => Err(format!("cannot parse '{s}' as Example")),
        }
    }
}

impl Example {
    fn paths(self) -> io::Result<ExamplePaths> {
        let base_dir = lace_data_dir().map(|dir| dir.join(self.to_str()))?;
        Ok(ExamplePaths {
            data: base_dir.join("data.csv"),
            codebook: base_dir.join("codebook.yaml"),
            lace: base_dir.join("lace"),
        })
    }

    pub fn regen_metadata(
        self,
        n_iters: usize,
        timeout: Option<u64>,
    ) -> Result<(), Error> {
        use crate::config::EngineUpdateConfig;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256Plus;

        let n_states = 16;

        let paths = self.paths()?;
        // Delete the metadata entirely. This will get rid of extra states and
        // diagnostics that we might have if we've previously worked on a branch
        // with a higher number of example states.
        {
            let lacefile = paths.lace.as_path();
            // Can't remove a directory that doesn't exist
            if lacefile.exists() {
                std::fs::remove_dir_all(lacefile)?;
                std::fs::create_dir(lacefile)?;
            }
        }

        let codebook: Codebook = {
            let mut file = std::fs::File::open(&paths.codebook)?;
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_yaml::from_str(ser.as_str()).map_err(|err| {
                eprint!("{err:?}");
                let err_kind = io::ErrorKind::InvalidData;
                io::Error::new(err_kind, "Could not parse codebook")
            })?
        };

        let mut engine = crate::Engine::new(
            n_states,
            codebook,
            DataSource::Csv(paths.data),
            0,
            Xoshiro256Plus::seed_from_u64(1337),
        )
        .map_err(|_| {
            let err_kind = io::ErrorKind::Other;
            io::Error::new(err_kind, "Failed to create Engine")
        })?;

        let config = EngineUpdateConfig::new()
            .default_transitions()
            .n_iters(n_iters);

        let timeout = Timeout::new(
            timeout.map(Duration::from_secs).unwrap_or(Duration::MAX),
        );

        engine.update(&config, timeout)?;
        engine
            .save(paths.lace.as_path(), lace_metadata::SerializedType::Yaml)?;
        Ok(())
    }

    /// Get an oracle build for the example. If this is the first time using
    /// the example, a new analysis will run. Be patient.
    pub fn oracle(self) -> Result<Oracle, Error> {
        let paths = self.paths()?;
        if !paths.lace.exists() {
            self.regen_metadata(DEFAULT_N_ITERS, DEFAULT_TIMEOUT)?;
        }
        Oracle::load(paths.lace.as_path())
    }

    /// Get an engine build for the example. If this is the first time using
    /// the example, a new analysis will run. Be patient.
    pub fn engine(self) -> Result<Engine, Error> {
        let paths = self.paths()?;
        if !paths.lace.exists() {
            self.regen_metadata(DEFAULT_N_ITERS, DEFAULT_TIMEOUT)?;
        }
        Engine::load(paths.lace.as_path())
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_str(&self) -> &str {
        match self {
            Example::Animals => "animals",
            Example::Satellites => "satellites",
        }
    }
}

/// Creates and returns a data directory
fn lace_data_dir() -> io::Result<PathBuf> {
    let data_dir: PathBuf = dirs::data_dir()
        .map(|dir| dir.join("lace").join("examples"))
        .ok_or({
            let err_kind = io::ErrorKind::NotFound;
            io::Error::new(err_kind, "could not find user data directory")
        })?;

    create_dir_all(&data_dir).map(|_| data_dir)
}

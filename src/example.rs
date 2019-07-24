use std::fs::create_dir_all;
use std::io::{self, Read};
use std::path::PathBuf;
use braid_codebook::codebook::Codebook;
use crate::data::DataSource;
use crate::{Oracle, Engine, EngineBuilder};

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
/// # use braid::example::Example;
/// let oracle = Example::Animals.oracle().unwrap();
/// 
/// assert!(oracle.rowsim(31, 32, None) < oracle.rowsim(33, 32, None));
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Example {
    /// A dataset with animals and their features
    Animals,
}

impl Example {
    fn paths(&self) -> io::Result<ExamplePaths> {
        let base_dir = braid_data_dir().map(|dir| dir.join(self.to_str()))?;
        Ok(ExamplePaths {
            data: base_dir.join("data.csv"),
            codebook: base_dir.join("codebook.yaml"),
            braid: base_dir.join("braid"),
        })
    }

    /// Get an oracle build for the example. If this is the first time using
    /// the example, a new analysis will run. Be patient.
    pub fn oracle(&self) -> io::Result<Oracle> {
        let paths = self.paths()?;
        if paths.braid.exists() {
            Oracle::load(paths.braid.as_path())
        } else {
            let codebook: Codebook = {
                let mut file = std::fs::File::open(&paths.codebook)?;
                let mut ser = String::new();
                file.read_to_string(&mut ser)?;
                serde_yaml::from_str(&ser.as_str()).map_err(|_| {
                    let err_kind = io::ErrorKind::InvalidData;
                    io::Error::new(err_kind, "Could not parse codebook")
                })?
            };

            let mut engine: Engine = EngineBuilder::new(DataSource::Csv(paths.data))
                .with_codebook(codebook)
                .with_nstates(8)
                .build()
                .map_err(|_| {
                    let err_kind = io::ErrorKind::Other;
                    io::Error::new(err_kind, "Failed to create Engine")
                })?;

            engine.run(500);
            engine.save_to(&paths.braid.as_path()).save()?;

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

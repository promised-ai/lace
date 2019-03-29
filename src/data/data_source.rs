extern crate braid_codebook;
extern crate csv;

use std::path::Path;

use braid_codebook::codebook::Codebook;
use braid_codebook::csv::codebook_from_csv;
use csv::ReaderBuilder;

use crate::result;

/// Denotes the source type of the data to be analyzed
#[derive(Debug, Clone)]
pub enum DataSource {
    Sqlite(String),
    Postgres(String),
    Csv(String),
}

impl DataSource {
    /// Return a `Path` to the source
    pub fn to_path(&self) -> &Path {
        match &self {
            DataSource::Sqlite(s) => Path::new(s),
            DataSource::Postgres(s) => Path::new(s),
            DataSource::Csv(s) => Path::new(s),
        }
    }

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> result::Result<Codebook> {
        match &self {
            DataSource::Csv(s) => {
                let csv_reader = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(Path::new(s.as_str()))?;
                Ok(codebook_from_csv(csv_reader, None, None, None))
            }
            _ => {
                let msg =
                    format!("Default codebook for {:?} not implemented", &self);
                Err(result::Error::new(
                    result::ErrorKind::NotImplementedError,
                    msg.as_str(),
                ))
            }
        }
    }
}

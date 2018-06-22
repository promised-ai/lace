extern crate csv;

use self::csv::ReaderBuilder;
use cc::Codebook;
use data::csv::codebook_from_csv;
use std::io;
use std::path::Path;

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
    pub fn default_codebook(&self) -> io::Result<Codebook> {
        match &self {
            DataSource::Csv(s) => {
                let mut csv_reader = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(Path::new(s.as_str()))?;
                Ok(codebook_from_csv(csv_reader, None, None))
            }
            _ => {
                let msg =
                    format!("Default codebook for {:?} not implemented", &self);
                let kind = io::ErrorKind::InvalidInput;
                let err = io::Error::new(kind, msg.as_str());
                Err(err)
            }
        }
    }
}

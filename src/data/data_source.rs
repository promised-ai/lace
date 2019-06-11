use std::{convert::From, ffi::OsString, path::PathBuf};

use braid_codebook::{codebook::Codebook, csv::codebook_from_csv};
use csv::ReaderBuilder;

use crate::result;

/// Denotes the source type of the data to be analyzed
#[derive(Debug, Clone)]
pub enum DataSource {
    /// SQLite database
    Sqlite(PathBuf),
    /// Postgres database
    Postgres(PathBuf),
    /// CSV file
    Csv(PathBuf),
}

impl From<DataSource> for PathBuf {
    fn from(src: DataSource) -> PathBuf {
        match src {
            DataSource::Sqlite(s) => s,
            DataSource::Postgres(s) => s,
            DataSource::Csv(s) => s,
        }
    }
}

impl DataSource {
    pub fn to_os_string(&self) -> OsString {
        match self {
            DataSource::Sqlite(s) => s,
            DataSource::Postgres(s) => s,
            DataSource::Csv(s) => s,
        }
        .clone()
        .into_os_string()
    }

    pub fn to_string(&self) -> String {
        self.to_os_string().into_string().unwrap()
    }

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> result::Result<Codebook> {
        match &self {
            DataSource::Csv(s) => {
                let csv_reader =
                    ReaderBuilder::new().has_headers(true).from_path(s)?;
                Ok(codebook_from_csv(csv_reader, None, None, None))
            }
            _ => {
                let msg =
                    format!("Default codebook for {:?} not implemented", &self);
                Err(result::Error::new(
                    result::ErrorKind::NotImplementedError,
                    msg,
                ))
            }
        }
    }
}

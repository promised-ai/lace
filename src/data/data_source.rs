//! Type of the data source, e.g., CSV or SQL database.
use std::convert::From;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

use braid_codebook::codebook::Codebook;
use braid_codebook::csv::codebook_from_csv;
use csv::ReaderBuilder;

use super::error::data_source::DefaultCodebookError;

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

impl fmt::Display for DataSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_os_string().into_string().unwrap())
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

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> Result<Codebook, DefaultCodebookError> {
        match &self {
            DataSource::Csv(s) => ReaderBuilder::new()
                .has_headers(true)
                .from_path(s)
                .map_err(|_| DefaultCodebookError::DataNotFoundError)
                .map(|csv_reader| {
                    codebook_from_csv(csv_reader, None, None, None)
                }),
            _ => Err(DefaultCodebookError::UnsupportedDataSrouceError),
        }
    }
}

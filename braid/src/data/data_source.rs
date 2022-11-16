//! Type of the data source, e.g., CSV or SQL database.
use super::error::DefaultCodebookError;
use braid_codebook::Codebook;
use polars::prelude::json::write::Empty;
use std::convert::TryFrom;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

/// Denotes the source type of the data to be analyzed
#[derive(Debug, Clone)]
pub enum DataSource {
    /// CSV file
    Csv(PathBuf),
    /// Apache IPC data format (e.g. Feather V2)
    Ipc(PathBuf),
    /// JSON  or JSON line file
    Json(PathBuf),
    /// Parquet data format
    Parquet(PathBuf),
    /// Empty (A void datasource).
    Empty,
}

impl TryFrom<DataSource> for PathBuf {
    type Error = &'static str;
    fn try_from(src: DataSource) -> Result<Self, Self::Error> {
        match src {
            DataSource::Parquet(s)
            | DataSource::Csv(s)
            | DataSource::Json(s)
            | DataSource::Ipc(s) => Ok(s),
            DataSource::Empty => {
                Err("DataSource::EMPTY has no path information")
            }
        }
    }
}

impl fmt::Display for DataSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            self.to_os_string()
                .and_then(|s| s.into_string().ok())
                .unwrap_or_else(|| "EMPTY".to_owned())
        )
    }
}

impl DataSource {
    pub fn to_os_string(&self) -> Option<OsString> {
        match self {
            DataSource::Parquet(s)
            | DataSource::Csv(s)
            | DataSource::Json(s)
            | DataSource::Ipc(s) => Some(s),
            DataSource::Empty => None,
        }
        .map(|x| x.clone().into_os_string())
    }

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> Result<Codebook, DefaultCodebookError> {
        use crate::codebook::parquet;
        match &self {
            DataSource::Ipc(path) => {
                Ok(parquet::codebook_from_ipc(path, None, None))
            }
            DataSource::Csv(path) => {
                Ok(parquet::codebook_from_csv(path, None, None))
            }
            DataSource::Json(path) => {
                Ok(parquet::codebook_from_json(path, None, None))
            }
            DataSource::Parquet(path) => {
                Ok(parquet::codebook_from_parquet(path, None, None))
            }
            DataSource::Empty => Ok(Codebook::default()),
        }
    }
}

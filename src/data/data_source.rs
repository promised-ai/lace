//! Type of the data source, e.g., CSV or SQL database.
use std::convert::TryFrom;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

use braid_codebook::csv::{codebook_from_csv, ReaderGenerator};
use braid_codebook::{Codebook, ColMetadataList};

use super::error::DefaultCodebookError;

/// Denotes the source type of the data to be analyzed
#[derive(Debug, Clone)]
pub enum DataSource {
    /// Postgres database
    Postgres(PathBuf),
    /// CSV file
    Csv(PathBuf),
    /// GZipped CSV
    GzipCsv(PathBuf),
    /// Empty (A void datasource).
    Empty,
}

impl TryFrom<DataSource> for PathBuf {
    type Error = &'static str;
    fn try_from(src: DataSource) -> Result<Self, Self::Error> {
        match src {
            DataSource::Postgres(s)
            | DataSource::Csv(s)
            | DataSource::GzipCsv(s) => Ok(s),
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
            DataSource::Postgres(s)
            | DataSource::Csv(s)
            | DataSource::GzipCsv(s) => Some(s),
            DataSource::Empty => None,
        }
        .map(|x| x.clone().into_os_string())
    }

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> Result<Codebook, DefaultCodebookError> {
        match &self {
            DataSource::Csv(s) => {
                let generator = ReaderGenerator::Csv(s.to_owned());
                codebook_from_csv(generator, None, None, true).map_err(DefaultCodebookError::FromCsvError)
            },
            DataSource::GzipCsv(s) => {
                let generator = ReaderGenerator::GzipCsv(s.to_owned());
                codebook_from_csv(generator, None, None, true).map_err(DefaultCodebookError::FromCsvError)
            },
            DataSource::Empty => Ok(Codebook::new(
                "Empty".to_owned(),
                ColMetadataList::new(vec![]).unwrap(),
            )),
            _ => Err(DefaultCodebookError::UnsupportedDataSrouce),
        }
    }
}

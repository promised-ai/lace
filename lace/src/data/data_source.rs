//! Type of the data source, e.g., CSV or SQL database.
use super::error::DefaultCodebookError;
use lace_codebook::Codebook;
use polars::frame::DataFrame;
use std::fmt;

#[cfg(feature = "formats")]
use std::ffi::{OsStr, OsString};
#[cfg(feature = "formats")]
use std::path::PathBuf;

/// Denotes the source type of the data to be analyzed
#[cfg(not(feature = "formats"))]
#[derive(Debug, Clone, PartialEq)]
pub enum DataSource {
    /// Polars DataFrame
    Polars(DataFrame),
    /// Empty (A void datasource).
    Empty,
}

/// Denotes the source type of the data to be analyzed
#[cfg(feature = "formats")]
#[derive(Debug, Clone, PartialEq)]
pub enum DataSource {
    /// CSV file
    Csv(PathBuf),
    /// Apache IPC data format (e.g. Feather V2)
    Ipc(PathBuf),
    /// JSON  or JSON line file
    Json(PathBuf),
    /// Parquet data format
    Parquet(PathBuf),
    /// Polars DataFrame
    Polars(DataFrame),
    /// Empty (A void datasource).
    Empty,
}

/// Error when extension is not CSV, JSON, IPC, or Parquet
#[cfg(feature = "formats")]
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownExtension(pub Option<OsString>);

#[cfg(feature = "formats")]
impl std::fmt::Display for UnknownExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown Extension: {:?}", self.0)
    }
}

#[cfg(feature = "formats")]
impl std::error::Error for UnknownExtension {}

#[cfg(feature = "formats")]
impl TryFrom<PathBuf> for DataSource {
    type Error = UnknownExtension;

    fn try_from(value: PathBuf) -> Result<Self, Self::Error> {
        match value
            .extension()
            .and_then(OsStr::to_str)
            .map(str::to_lowercase)
            .ok_or_else(|| {
                UnknownExtension(value.extension().map(OsStr::to_os_string))
            })?
            .as_ref()
        {
            "csv" | "csv.gz" => Ok(Self::Csv(value)),
            "gz" if value.ends_with("") => Ok(Self::Csv(value)),
            "json" | "jsonl" => Ok(Self::Json(value)),
            "parquet" => Ok(Self::Parquet(value)),
            "arrow" | "ipc" => Ok(Self::Ipc(value)),
            _ => Err(UnknownExtension(
                value.extension().map(OsStr::to_os_string),
            )),
        }
    }
}

#[cfg(feature = "formats")]
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
            DataSource::Polars(_) => {
                Err("DataSource::Polars has no corresponding path")
            }
        }
    }
}

impl From<DataFrame> for DataSource {
    fn from(value: DataFrame) -> Self {
        Self::Polars(value)
    }
}

#[cfg(feature = "formats")]
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

#[cfg(not(feature = "formats"))]
impl fmt::Display for DataSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Polars(df) => {
                write!(f, "polars::DataFrame {:?}", df.shape())
            }
            Self::Empty => {
                write!(f, "Empty")
            }
        }
    }
}

#[cfg(feature = "formats")]
impl DataSource {
    pub fn to_os_string(&self) -> Option<OsString> {
        match self {
            DataSource::Parquet(s)
            | DataSource::Csv(s)
            | DataSource::Json(s)
            | DataSource::Ipc(s) => Some(s),
            DataSource::Empty | DataSource::Polars(_) => None,
        }
        .map(|x| x.clone().into_os_string())
    }

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> Result<Codebook, DefaultCodebookError> {
        use crate::codebook::{data, formats};
        let codebook = match &self {
            DataSource::Ipc(path) => {
                formats::codebook_from_ipc(path, None, None, false)
            }
            DataSource::Csv(path) => {
                formats::codebook_from_csv(path, None, None, false)
            }
            DataSource::Json(path) => {
                formats::codebook_from_json(path, None, None, false)
            }
            DataSource::Parquet(path) => {
                formats::codebook_from_parquet(path, None, None, false)
            }
            DataSource::Polars(df) => {
                data::df_to_codebook(df, None, None, false)
            }
            DataSource::Empty => Ok(Codebook::default()),
        }?;
        Ok(codebook)
    }
}

#[cfg(not(feature = "formats"))]
impl DataSource {
    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> Result<Codebook, DefaultCodebookError> {
        use crate::codebook::data;
        let codebook = match &self {
            DataSource::Polars(df) => {
                data::df_to_codebook(df, None, None, false)
            }
            DataSource::Empty => Ok(Codebook::default()),
        }?;
        Ok(codebook)
    }
}

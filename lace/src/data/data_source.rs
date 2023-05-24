//! Type of the data source, e.g., CSV or SQL database.
use super::error::DefaultCodebookError;
use lace_codebook::Codebook;
use polars::frame::DataFrame;
use std::convert::TryFrom;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;

/// Denotes the source type of the data to be analyzed
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
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownExtension(pub Option<OsString>);

impl std::fmt::Display for UnknownExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown Extension: {:?}", self.0)
    }
}

impl std::error::Error for UnknownExtension {}

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

impl From<DataFrame> for DataSource {
    fn from(value: DataFrame) -> Self {
        Self::Polars(value)
    }
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
            DataSource::Polars(_) => {
                Err("DataSource::Polars has no corresponding path")
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
            DataSource::Empty | DataSource::Polars(_) => None,
        }
        .map(|x| x.clone().into_os_string())
    }

    /// Generate a default `Codebook` from the source data
    pub fn default_codebook(&self) -> Result<Codebook, DefaultCodebookError> {
        use crate::codebook::data;
        let codebook = match &self {
            DataSource::Ipc(path) => {
                data::codebook_from_ipc(path, None, None, false)
            }
            DataSource::Csv(path) => {
                data::codebook_from_csv(path, None, None, false)
            }
            DataSource::Json(path) => {
                data::codebook_from_json(path, None, None, false)
            }
            DataSource::Parquet(path) => {
                data::codebook_from_parquet(path, None, None, false)
            }
            DataSource::Polars(df) => {
                data::df_to_codebook(df, None, None, false)
            }
            DataSource::Empty => Ok(Codebook::default()),
        }?;
        Ok(codebook)
    }
}

#[cfg(test)]
mod tests {
    use super::{DataSource, UnknownExtension};
    use std::path::PathBuf;

    #[test]
    fn data_source_from_path() {
        macro_rules! pathcheck_ok {
            ($path: literal, $expected: ident) => {
                let path = PathBuf::from($path);
                let maybe_datasource = path.clone().try_into();
                assert_eq!(maybe_datasource, Ok(DataSource::$expected(path)));
            };
        }

        macro_rules! pathcheck_err {
            ($path: literal, $expected: expr) => {
                let path = PathBuf::from($path);
                let maybe_datasource: Result<DataSource, UnknownExtension> =
                    path.clone().try_into();
                assert_eq!(
                    maybe_datasource,
                    Err(UnknownExtension($expected.map(|x: &str| x.into())))
                );
            };
        }

        pathcheck_ok!("./abc.csv", Csv);
        pathcheck_ok!("/xyz/efg.hji//abc.csv", Csv);
        pathcheck_ok!("./abc.csv.gz", Csv);
        pathcheck_ok!("./abc.json", Json);
        pathcheck_ok!("./abc.jsonl", Json);
        pathcheck_ok!("./abc.parquet", Parquet);
        pathcheck_ok!("./abc.arrow", Ipc);
        pathcheck_ok!("./abc.ipc", Ipc);

        pathcheck_err!("./abc.123", Some("123"));
        pathcheck_err!("./abc", None);
    }
}

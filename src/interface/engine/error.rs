use crate::data::CsvParseError;
use serde::Serialize;

/// Errors that can arise when parsing data for an Engine
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataParseError {
    IoError,
    /// Problem converting a Sqlite table into an Engine
    SqliteError,
    /// Problem converting a Postgres table into an Engine
    PostgresError,
    /// Problem parsing the input CSV into an Engine
    CsvParseError(CsvParseError),
    /// The supplied data source is not currently supported for this operation
    UnsupportedDataSourceError,
}

impl Into<DataParseError> for CsvParseError {
    fn into(self) -> DataParseError {
        match self {
            CsvParseError::IoError => DataParseError::IoError,
            _ => DataParseError::CsvParseError(self),
        }
    }
}

/// Errors that can arise when creating a new engine
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum NewEngineError {
    /// Asked for zero states. The Engine must have at least one state.
    ZeroStatesRequestedError,
    /// Problem parsing the input data into an Engine
    DataParseError(DataParseError),
}

/// Errors that can arise when appending new features to an Engine
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum InsertDataError {
    NoRowNamesInCodebookError,
    NoColumnMetadataError,
    TooManyEntriesInColumnMetadataError,
    NewColumnNotInColumnMetadataError(String),
    ModeForbidsOverwriteError,
    ModeForbidsNewRowsError,
    ModeForbidsNewColumnsError,
    ModeForbidsNewRowsOrColumnsError,
    NoGaussianHyperForNewColumn(String),
    DatumIncompatibleWithColumn(String),
}

use crate::cc::FType;
use crate::data::CsvParseError;
use std::io;
use thiserror::Error;

/// Errors that can arise when parsing data for an Engine
#[derive(Debug, Error)]
pub enum DataParseError {
    /// Problem deriving from the `csv` crate
    #[error("csv error: {0}")]
    CsvError(#[from] csv::Error),
    /// Problem reading the file
    #[error("io error: {0}")]
    IoError(#[from] io::Error),
    /// Problem converting a Sqlite table into an Engine
    #[error("sqlite error")]
    SqliteError,
    /// Problem converting a Postgres table into an Engine
    #[error("postgres error")]
    PostgresError,
    /// Problem parsing the input CSV into an Engine
    #[error("csv parse error: {0}")]
    CsvParseError(#[from] CsvParseError),
    /// The supplied data source is not currently supported for this operation
    #[error("Provided an unsupported data source")]
    UnsupportedDataSource,
}

/// Errors that can arise when creating a new engine
#[derive(Debug, Error)]
pub enum NewEngineError {
    /// Asked for zero states. The Engine must have at least one state.
    #[error("attempted to create an engine with zero states")]
    ZeroStatesRequested,
    /// Problem parsing the input data into an Engine
    #[error("data parse error: {0}")]
    DataParseError(DataParseError),
}

/// Errors that can arise when appending new features to an Engine
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InsertDataError {
    /// Missing column metadata for a column
    #[error("No column metadata for column '{0}'")]
    NoColumnMetadataForColumn(String),
    /// There should be the same number of entries in column_metadata as there
    /// are new columns to append
    #[error("the number of entries in col_metadata must match the number of new columns ({ncolmd} != {nnew})")]
    WrongNumberOfColumnMetadataEntries {
        /// number of entries in supplied column metadata
        ncolmd: usize,
        /// Number of new columns to append
        nnew: usize,
    },
    /// a column is missing from the metadata
    #[error("The new column '{0}' was not found in the metadata")]
    NewColumnNotInColumnMetadata(String),
    /// the insert mode does not allow overwriting
    #[error("Overwrite forbidden with requested mode")]
    ModeForbidsOverwrite,
    /// the insert mode does not allow new rows
    #[error("New rows forbidden with requested mode")]
    ModeForbidsNewRows,
    /// the insert mode does not allow new columns
    #[error("New columns forbidden with requested mode")]
    ModeForbidsNewColumns,
    /// the insert mode does not allow new rows or columns
    #[error("New rows and columns forbidden with requested mode")]
    ModeForbidsNewRowsOrColumns,
    /// There was no hyper prior supplied for the Gaussian column
    #[error("No Gaussian hyper prior for new column '{0}'")]
    NoGaussianHyperForNewColumn(String),
    /// There was no hyper prior supplied for the Poisson column
    #[error("No Poisson hyper prior for new column '{0}'")]
    NoPoissonHyperForNewColumn(String),
    #[error(
        "Provided a {ftype_req:?} data for '{col}' but '{col}' is {ftype:?}"
    )]
    DatumIncompatibleWithColumn {
        col: String,
        ftype_req: FType,
        ftype: FType,
    },
    /// Tried to add a row with no values in it
    #[error("The row '{0}' is entirely empty")]
    EmptyRow(String),
}

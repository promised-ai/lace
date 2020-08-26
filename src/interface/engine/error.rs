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
    /// Problem converting a Postgres table into an Engine
    #[error("postgres error")]
    PostgresError,
    /// Problem parsing the input CSV into an Engine
    #[error("csv parse error: {0}")]
    CsvParseError(#[from] CsvParseError),
    /// The supplied data source is not currently supported for this operation
    #[error("Provided an unsupported data source")]
    UnsupportedDataSource,
    /// The user supplied column_metdata in the codebook but provided an empty
    /// data source
    #[error("non-empty column_metdata the codebook but empty DataStouce")]
    ColumnMetadataSuppliedForEmptyData,
    /// The user supplied row_names in the codebook but provided an empty
    /// data source
    #[error("non-empty row_names the codebook but empty DataStouce")]
    RowNamesSuppliedForEmptyData,
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
    #[error(
        "the number of entries in col_metadata must match the number of new \
         columns ({ncolmd} != {nnew})"
    )]
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
    #[error("Overwrite forbidden by requested mode")]
    ModeForbidsOverwrite,
    /// the insert mode does not allow new rows
    #[error("New rows forbidden by requested mode")]
    ModeForbidsNewRows,
    /// the insert mode does not allow new columns
    #[error("New columns forbidden by requested mode")]
    ModeForbidsNewColumns,
    /// the insert mode does not allow new rows or columns
    #[error("New rows and columns forbidden by requested mode")]
    ModeForbidsNewRowsOrColumns,
    /// the insert mode does not allow the extension of categorical column
    /// cardinalities.
    #[error(
        "Categorical column support extension forbidden by requested mode"
    )]
    ModeForbidsCategoryExtension,
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
    /// No metdata was supplied for a categorical column whose support we
    /// wished to extend
    #[error(
        "No insert col_metadata supplied for '{col_name}'. Categorical column \
        '{col_name}' has a value_map, so to extend k from {ncats} to \
        {ncats_req}, a value_map must be supplied in col_metadata to add the \
        new values and maintain a valid codebook"
    )]
    NoNewValueMapForCategoricalExtension {
        ncats: usize,
        ncats_req: usize,
        col_name: String,
    },
    /// The insert operation requires a column metadata, but the wrong metadata
    /// for that column contains the wrong `ColType`
    #[error(
        "Passed {ftype_md:?} ColType through col_metadata for column \
         {col_name}, which is {ftype:?}"
    )]
    WrongMetadataColType {
        col_name: String,
        ftype: FType,
        ftype_md: FType,
    },
    /// The insert operation requires a value map be supplied by the user under
    /// a entry in the `col_metadata` argument, but the supplied value_map is
    /// incompatible with the requested operation. For example, the user is
    /// adding a category to a categorical column with a value map but the
    /// supplied value map does not cover one or more of the existing categories
    /// or one or more of the new categories.
    #[error(
        "The value_map supplied for column {col_name} does not contain the \
         correct entries to support the requested operation."
    )]
    IncompleteValueMap { col_name: String },
}

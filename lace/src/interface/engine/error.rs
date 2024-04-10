use std::io;

use crate::error::IndexError;
use lace_cc::feature::FType;
use lace_codebook::CodebookError;
use lace_data::Category;
use thiserror::Error;

/// Errors that can arise when parsing data for an Engine
#[derive(Debug, Error)]
pub enum DataParseError {
    /// Problem reading the file
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    /// Problem parsing the input CSV into an Engine
    #[error("Codebook error: {0}")]
    Codebook(#[from] CodebookError),
    /// The supplied data source is not currently supported for this operation
    #[error("Provided an unsupported data source")]
    UnsupportedDataSource,
    /// The user supplied column_metadata in the codebook but provided an empty
    /// data source
    #[error("non-empty column_metadata the codebook but empty DataSource")]
    ColumnMetadataSuppliedForEmptyData,
    /// The user supplied row_names in the codebook but provided an empty
    /// data source
    #[error("non-empty row_names the codebook but empty DataSource")]
    RowNamesSuppliedForEmptyData,
    /// There is no `ID` column in the dataset
    #[error("No 'ID' column")]
    NoIDColumn,
    /// There is more than one ID column
    #[error("Multiple ID columns: {0:?}")]
    MultipleIdColumns(Vec<String>),
    /// There is a column type in the codebook that is not supported for loading
    /// externally
    #[error("Column `{col_name}` has type `{col_type}`, which is unsupported for external data sources")]
    UnsupportedColumnType { col_name: String, col_type: String },
    /// The codebook and the data have a different number of rows
    #[error("The codebook contains {n_codebook_rows} rows, but the data contain {n_data_rows} rows")]
    CodebookAndDataRowsMismatch {
        n_codebook_rows: usize,
        n_data_rows: usize,
    },
    #[error(
        "The dataframe does not contain the column `{column}` listed in the codebook"
    )]
    DataFrameMissingColumn { column: String },
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

/// Errors that can arise when inserting data into the Engine
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
    #[error("No Categorical hyper prior for new column '{0}'")]
    NoCategoricalHyperForNewColumn(String),
    #[error("No StickBreakingDiscrete hyper prior for new column '{0}'")]
    NoStickBreakingDiscreteHyperForNewColumn(String),
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
    /// No metadata was supplied for a categorical column whose support we
    /// wished to extend
    #[error(
        "No insert col_metadata supplied for '{col_name}'. Categorical column \
        '{col_name}' has a value_map, so to extend k from {n_cats} to \
        {n_cats_req}, a value_map must be supplied in col_metadata to add the \
        new values and maintain a valid codebook"
    )]
    NoNewValueMapForCategoricalExtension {
        n_cats: usize,
        n_cats_req: usize,
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
    /// The user tried to insert a NaN, -Inf, or Inf into a continuous column
    #[error(
        "Attempted to insert a non-finite value ({value}) into column \
        `{col}`"
    )]
    NonFiniteContinuousValue { col: String, value: f64 },
    #[error("Row index error: {0}")]
    RowIndex(IndexError),
    #[error("Column index error: {0}")]
    ColumnIndex(IndexError),
    /// An placeholder error variant used when chaining `ok_or` with `map_or`
    #[error("How can you extract what is unreachable?")]
    Unreachable,
    #[error(
        "The column with usize index '{0}' appears to be new, but new columns \
        must be given string names"
    )]
    IntegerIndexNewColumn(usize),
    #[error(
        "The row with usize index '{0}' appears to be new, but new rows \
        must be given string names"
    )]
    IntegerIndexNewRow(usize),
    #[error("Tried to extend to support of boolean column '{0}'")]
    ExtendBooleanColumn(String),
    #[error("Could not find value in categorical value map")]
    CategoryNotInValueMap(Category),
    #[error("Attempted to add a category '{1}' to a column of type '{0}' for column '{2}'")]
    WrongCategoryAndType(String, String, String),
}

/// Errors that can arise when removing data from the engine
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RemoveDataError {
    /// The requested index does not exist
    #[error("Index error: {0}")]
    Index(#[from] IndexError),
}

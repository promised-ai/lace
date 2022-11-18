use polars::datatypes::DataType;
use serde::Serialize;
use thiserror::Error;

/// Error that can occur when merging the columns of two codebooks
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Error)]
pub enum MergeColumnsError {
    /// The two codebooks have overlapping column names
    #[error("The codebooks both have entries for column `{0}`")]
    DuplicateColumnName(String),
}

/// The row already exists
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("The row `{0}` alread exists")]
pub struct InsertRowError(pub String);

#[derive(Error, Debug)]
pub enum ReadError {
    #[error("Io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] polars::prelude::PolarsError),
}

#[derive(Debug, Error)]
pub enum ColMetadataListError {
    #[error("Duplicate column name `{0}`")]
    Duplicate(String),
}

#[derive(Debug, Error)]
pub enum RowNameListError {
    #[error("Duplicate row name `{row_name}` at index {ix_1} and {ix_1}")]
    Duplicate {
        row_name: String,
        ix_1: usize,
        ix_2: usize,
    },
}

/// Errors that can arise when creating a codebook from a CSV file
#[derive(Debug, Error)]
pub enum CodebookError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A column had no values in it.
    #[error("column '{col_name}' is blank")]
    BlankColumn { col_name: String },
    /// Could not infer the data type of a column
    #[error("cannot infer feature type of column '{col_name}'")]
    UnableToInferColumnType { col_name: String },
    /// Dataset contains a Arrow datatype that cannot currently be converted to
    /// a column metadata
    #[error("Unsupported Arrow dtype `{dtype}` for column `{col_name}`")]
    UnsupportedDataType { col_name: String, dtype: DataType },
    /// Too many distinct values for categorical column. A category is
    /// represented by a u8, so there can only be 256 distinct values.
    #[error("column '{col_name}' contains more than 256 categorical classes")]
    CategoricalOverflow { col_name: String },
    /// The column with name appears more than once
    #[error("Column metadata error")]
    ColumnMetadata(#[from] ColMetadataListError),
    /// The column with name appears more than once
    #[error("Row names error: {0}")]
    RowNames(#[from] RowNameListError),
    /// Polars error
    #[error("Polars error: {0}")]
    Polars(#[from] polars::prelude::PolarsError),
    /// The user did not provide an index/ID column
    #[error("No `ID` column (row index)")]
    NoIdColumn,
    /// There are null values in the index/ID column
    #[error("Null values in ID column (row index)")]
    NullValuesInIndex,
    /// There the column contains only a single unique value, which can cause
    /// zero-variance issues which in turn cause other numerical issues.
    #[error("Column `{0}` contains only a single unique value")]
    SingleValueColumn(String),
    /// There is more than one column named some form of `ID`. For example,
    /// there is a column named `ID` and `id`.
    #[error("More than one `ID` column.")]
    MultipleIdColumns,
    /// Problem reading data into a DataFrame
    #[error("ReadError: {0}")]
    Read(#[from] ReadError),
}

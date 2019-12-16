use crate::data::CsvParseError;
use braid_codebook::MergeColumnsError;
use serde::Serialize;

/// Errors that can arise when appending new features to an Engine
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppendFeaturesError {
    IoError,
    /// A feature with this name already exists
    ColumnAlreadyExistsError(String),
    /// There is a mismatch between the feature names in the partial codebook
    /// and the feature names in the supplied data
    CodebookDataColumnNameMismatchError,
    /// The column lengths in the data source differ
    ColumnLengthError,
    /// Either there are rows in the new columns that have different names or
    /// they are not in the same order as the rows in the parent data
    RowNameMismatchError,
    /// There are no row names in the parent codebook, so the `CheckNames`
    /// `RowAlignmentStrategy` cannot be used.
    NoRowNamesInParentError,
    /// There are no row names in the child codebook, so the `CheckNames`
    /// `RowAlignmentStrategy` cannot be used.
    NoRowNamesInChildError,
    /// There are no row names in either codebook, so the `CheckNames`
    /// `RowAlignmentStrategy` cannot be used.
    NoRowNamesError,
    /// Problem parsing the data
    DataParseError(DataParseError),
}

impl Into<AppendFeaturesError> for MergeColumnsError {
    fn into(self) -> AppendFeaturesError {
        match self {
            MergeColumnsError::DuplicateColumnNameError(name) => {
                AppendFeaturesError::ColumnAlreadyExistsError(name)
            }
        }
    }
}

/// Errors that can arise when appending new rows to an Engine
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppendRowsError {
    IoError,
    /// The feature data source is not supported
    UnsupportedDataSourceError,
    /// The row lengths in the data source differ. The new rows must contain
    /// entries for each column.
    RowLengthMismatchError,
    /// The number of entries in columns of the new rows differs
    ColumLengthMismatchError,
    /// Problem parsing the CSV of new rows
    DataParseError(DataParseError),
}

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
    NoPartialCodebookError,
    TooManyEntriesInPartialCodebookError,
    NewColumnNotInPartialCodebookError(String),
    ModeForbidsOverwriteError,
    ModeForbidsNewRowsError,
    ModeForbidsNewColumnsError,
    ModeForbidsNewRowsOrColumnsError,
    NoGaussianHyperForNewColumn(String),
    DatumIncompatibleWithColumn(String),
}

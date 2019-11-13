use crate::data::CsvParseError;
use braid_codebook::error::MergeColumnsError;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppendFeaturesError {
    /// The feature data source is not supported
    UnsupportedDataSourceError,
    /// A feature with this name already exists
    ColumnAlreadyExistsError(String),
    /// There is a mismatch between the feature names in the partial codebook
    /// and the feature names in the supplied data
    CodebookDataColumnNameMismatchError,
    /// The column lengths in the data source differ
    NewColumnLengthError,
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppendRowsError {
    /// The feature data source is not supported
    UnsupportedDataSourceError,
    /// The row lengths in the data source differ. The new rows must contain
    /// entries for each column.
    RowLengthMismatchError,
    /// The number of entries in columns of the new rows differs
    ColumLengthMismatchError,
    /// Problem parsing the CSV of new rows
    CsvParseError(CsvParseError),
}

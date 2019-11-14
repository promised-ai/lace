use serde::Serialize;

/// Error that can occur when merging the columns of two codebooks
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum MergeColumnsError {
    /// The two codebooks have overlapping column names
    DuplicateColumnNameError(String),
}

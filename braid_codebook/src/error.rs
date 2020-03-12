use serde::Serialize;
use std::error::Error;
use std::fmt;

/// Error that can occur when merging the columns of two codebooks
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub enum MergeColumnsError {
    /// The two codebooks have overlapping column names
    DuplicateColumnName(String),
}

/// The row already exists
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InsertRowError(pub String);

impl Error for MergeColumnsError {}
impl Error for InsertRowError {}

impl fmt::Display for MergeColumnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateColumnName(col) => {
                write!(f, "Found duplicate column: '{}'", col)
            }
        }
    }
}

impl fmt::Display for InsertRowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Row '{}' already exists", self.0)
    }
}

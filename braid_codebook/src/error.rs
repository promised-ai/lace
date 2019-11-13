use serde::{Deserialize, Serialize};

/// Error that can occur when merging the columns of two codebooks
#[derive(
    Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
)]
pub enum MergeColumnsError {
    /// The two codebooks have overlapping column names
    DuplicateColumnNameError(String),
}

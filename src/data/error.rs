use crate::cc::FType;
use serde::{Deserialize, Serialize};

pub mod csv {
    use super::*;

    /// Errors that can arise while parsing a CSV together with a codebook
    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
    pub enum CsvParseError {
        /// The CSV file had no columns
        NoColumnsError,
        /// The first column must be named "ID" or "id"
        FirstColumnNotNamedIdError,
        /// There are one or more columns that are in the CSV, but not the codebook
        MissingCodebookColumnsError,
        /// There are one or more columns that are in the codebook but not the CSV
        MissingCsvColumnsError,
        /// The are duplicate columns in the CSV
        DuplicateCsvColumnsError,
        /// Could not parse the cell as the correct data type
        InvalidValueForColumnError {
            col_id: usize,
            row_name: String,
            val: String,
            col_type: FType,
        },
    }
}

pub mod data_source {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
    pub enum DefaultCodebookError {
        DataNotFoundError,
        UnsupportedDataSrouceError,
    }
}

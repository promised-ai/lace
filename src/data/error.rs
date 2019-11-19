use crate::cc::FType;
use serde::Serialize;

pub mod csv {
    use super::*;

    /// Errors that can arise while parsing a CSV together with a codebook
    #[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
    pub enum CsvParseError {
        /// Problem reading the file
        IoError,
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
        /// The columns of the csv and the columns in `codebook.col_metadata`
        /// must be in the same order
        CsvCodebookColumnsMisorderedError,
    }
}

pub mod data_source {
    use super::*;
    use braid_codebook::csv::FromCsvError;

    /// Errors that can arise generating the default codebook
    #[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
    pub enum DefaultCodebookError {
        /// Problem reading the data source
        IoError,
        /// The requested data source does not support default codebook
        /// generation
        UnsupportedDataSrouceError,
        /// Error deriving a codebook from a CSV
        FromCsvError(FromCsvError),
    }
}

use braid_cc::feature::FType;
use braid_codebook::csv::FromCsvError;
use std::io;
use thiserror::Error;

/// Errors that can arise while parsing a CSV together with a codebook
#[derive(Debug, Error)]
pub enum CsvParseError {
    /// Problem reading the file
    #[error("io error: {0}")]
    IoError(#[from] io::Error),
    /// The CSV file had no columns
    #[error("The csv contained no columns")]
    NoColumns,
    /// The first column must be named "ID" or "id"
    #[error("The first csv column must be named 'ID' or 'id'")]
    FirstColumnNotNamedId,
    /// There are one or more columns that are in the CSV, but not the codebook
    #[error("One or more columns appear in the csv that do not appear in the codebook")]
    MissingCodebookColumns,
    /// There are one or more columns that are in the codebook but not the CSV
    #[error("One or more columns appear in the codebook that do not appear in the csv")]
    MissingCsvColumns,
    /// There is a mismatch between the number of rows in the `row_names`
    /// codebook field and the number of rows in the data
    #[error("Different number of rows in codebook than in csv")]
    CodebookAndDataRowMismatch,
    /// The are duplicate columns in the CSV
    #[error("There are duplicate column names in the codebook")]
    DuplicateCodebookColumns,
    /// The are duplicate columns in the CSV
    #[error("There are duplicate column names in the csv")]
    DuplicateCsvColumns,
    /// The are duplicate row names in the CSV
    #[error("There are duplicate row names in the csv")]
    DuplicateCsvRows,
    /// Could not parse the cell as the correct data type
    #[error("Could not parse value '{val}' at row '{row_name}', column {col_id} into {col_type:?}")]
    InvalidValueForColumn {
        col_id: usize,
        row_name: String,
        val: String,
        col_type: FType,
    },
    /// The columns of the csv and the columns in `codebook.col_metadata`
    /// must be in the same order
    #[error("The columns of the csv and codebook must be in the same order")]
    CsvCodebookColumnsMisordered,
}

/// Errors that can arise generating the default codebook
#[derive(Debug, Error)]
pub enum DefaultCodebookError {
    /// Problem originating from the `csv` crate
    #[error("csv error: {0}")]
    CsvError(#[from] csv::Error),
    /// The requested data source does not support default codebook
    /// generation
    #[error("provided an unsupported data source")]
    UnsupportedDataSource,
    /// Error deriving a codebook from a CSV
    #[error("error generating codebook from csv: {0}")]
    FromCsvError(#[from] FromCsvError),
}

pub enum AppendFeaturesError {
    UnsupportedDataSourceError,
    CodebookDataColumnNameMismatchError,
    NewColumnLengthError,
}

pub enum AppendRowsError {
    UnsupportedDataSourceError,
    NewRowLengthError,
}

use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum StateLatentValuesError {
    #[error(
        "Attempted to propose latent values for non-latent columns: {0:?}"
    )]
    NonLatentColumns(Vec<usize>),
    /// The provide column index is out of bounds
    #[error("Asked for column index {col_ix} but there are {n_cols} columns")]
    ColumnIndexOutOfBounds { n_cols: usize, col_ix: usize },
    #[error("Asked for view index {view_ix}, but there are {n_views} views")]
    ViewIndexOutOfBounds { n_views: usize, view_ix: usize },
    #[error(
        "Incorrect number of rows in column {col_ix} values. View has \
        {view_rows} but values column has {values_rows}."
    )]
    ColumnLengthMismatch {
        col_ix: usize,
        view_rows: usize,
        values_rows: usize,
    },
}

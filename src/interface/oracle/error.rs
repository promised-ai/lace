//! Errors that can occur in Oracle functions
use braid_cc::feature::FType;
use thiserror::Error;

/// Describes errors arising from a bad `Given` in the context of an Oracle
/// query.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GivenError {
    /// The `Datum` for the column at `col_ix` is the wrong type, for example it
    /// was categorical when the column is continuous.
    #[error(
        "Provided {ftype_req:?} datum for column {col_ix}, which is {ftype:?}"
    )]
    InvalidDatumForColumn {
        /// The column index of the offending condition
        col_ix: usize,
        /// The FType of the Datum requested
        ftype_req: FType,
        /// The actual FType of the feature at col_ix
        ftype: FType,
    },
    /// The user passed a Datum::Missing a s a condition value
    #[error("Tried to condition on a 'missing' value in column {col_ix}")]
    MissingDatum { col_ix: usize },
    /// The column `col_ix` appears both in the `Given` and the target
    #[error("Column index {col_ix} appears in the target")]
    ColumnIndexAppearsInTarget { col_ix: usize },
    /// A column index in the given is out of bounds
    #[error("Index error in given: {0}")]
    IndexError(#[from] IndexError),
}

/// Describes errors that can occur from bad inputs to Oracle functions that
/// take indices are arguments
#[derive(Debug, Clone, PartialEq, Error)]
pub enum IndexError {
    /// The provide row index is out of bounds
    #[error("Asked for row index {row_ix} but there are {n_rows} rows")]
    RowIndexOutOfBounds { n_rows: usize, row_ix: usize },
    /// The provide column index is out of bounds
    #[error("Asked for column index {col_ix} but there are {n_cols} columns")]
    ColumnIndexOutOfBounds { n_cols: usize, col_ix: usize },
}

/// Errors that can occur from bad inputs to Oracle::rowsim
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RowSimError {
    /// One of the row indices is out of bounds or
    #[error(
        "Requested similarity for row {row_ix} but there are {n_rows} rows"
    )]
    RowIndexOutOfBounds { n_rows: usize, row_ix: usize },
    /// One of the column indices in wrt was out of bounds
    #[error("Requested wrt column {col_ix} but there are {n_cols} columns")]
    WrtColumnIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// The wrt was not `None`, but was an empty vector
    #[error("If wrt is not None, it must not be empty")]
    EmptyWrt,
}

/// Describes errors that can occur from bad inputs to `Oracle::mi`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MiError {
    /// Either or both of the column indices `col_a` or `col_b` is out of
    /// bounds
    #[error("Index error in 'mi' query: {0}")]
    IndexError(#[from] IndexError),
    /// The number of QMC samples requested is zero
    #[error("Must request more than zero samples")]
    NIsZero,
}

/// Describes errors that can occur from bad inputs to
/// `Oracle::conditional_entropy`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EntropyError {
    /// No target column indices provided
    #[error("No target columns provided")]
    NoTargetColumns,
    /// One or more of the target column indices is out of bounds
    #[error("Index error in entropy query: {0}")]
    IndexError(#[from] IndexError),
    /// The number of QMC samples requested is zero
    #[error("Must request more than zero samples")]
    NIsZero,
}

/// Describes errors that can occur from bad inputs to `Oracle::info_prop`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InfoPropError {
    /// No target column indices provided
    #[error("no target columns provided")]
    NoTargetColumns,
    /// No predictor column indices provided
    #[error("no predictor columns provided")]
    NoPredictorColumns,
    /// One or more of the target column indices is out of bounds
    #[error("target at index {col_ix} but there are {n_cols} columns")]
    TargetIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// One or more of the predictor column indices is out of bounds
    #[error("predictor at index {col_ix} but there are {n_cols} columns")]
    PredictorIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// The number of QMC samples requested is zero
    #[error("Must request more than zero samples")]
    NIsZero,
}

/// Describes errors that can occur from bad inputs to
/// `Oracle::conditional_entropy`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ConditionalEntropyError {
    /// One or more of the target column indices is out of bounds
    #[error("target at index {col_ix} but there are {n_cols} columns")]
    TargetIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// One or more of the predictor column indices is out of bounds
    #[error("predictor at index {col_ix} but there are {n_cols} columns")]
    PredictorIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// One or more predictor column indices occurs more than once
    #[error("predictor {col_ix} appears more than once")]
    DuplicatePredictors { col_ix: usize },
    /// No predictor columns provided
    #[error("no predictors provided")]
    NoPredictorColumns,
    /// The number of QMC samples requested is zero
    #[error("Must request more than zero samples")]
    NIsZero,
}

/// Describes errors that can occur from bad inputs to `Oracle::surprisal`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SurprisalError {
    /// One of the row or column indices is out of bounds
    #[error("Index error in surprisal query: {0}")]
    IndexError(#[from] IndexError),
    /// One or more of the optional state indices are out of bounds
    #[error("Requested state index {state_ix} but there are {nstates} states")]
    StateIndexOutOfBounds { nstates: usize, state_ix: usize },
    /// The `Datum` provided is incompatible with the requested column. Will
    /// not occur in `Oracle::self_surprisal`
    #[error(
        "Provided {ftype_req:?} datum for column {col_ix}, which is {ftype:?}"
    )]
    InvalidDatumForColumn {
        /// The column index of the offending condition
        col_ix: usize,
        /// The FType of the Datum requested
        ftype_req: FType,
        /// The actual FType of the feature at col_ix
        ftype: FType,
    },
}

/// Describes errors that can occur from bad inputs to `Oracle::predict`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PredictError {
    /// The target column index is out of bounds
    #[error("Target index error in predict query: {0}")]
    IndexError(#[from] IndexError),
    /// The Given is invalid
    #[error("Invalid predict 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum PredictUncertaintyError {
    /// The target column index is out of bounds
    #[error("Target index error in predict uncertainty query: {0}")]
    IndexError(#[from] IndexError),
    /// The Given is invalid
    #[error("Invalid predict uncertainty 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LogpError {
    /// No targets were supplies (empty vec)
    #[error("No target columns provided")]
    NoTargets,
    /// The number of values a row in `vals` does not equal the number of target
    /// indices in `col_ixs`
    #[error(
        "There are {ntargets} targets but a row in vals has {nvals} values"
    )]
    TargetsIndicesAndValuesMismatch { nvals: usize, ntargets: usize },
    /// The `Datum` for the target column at `col_ix` is the wrong type, for
    /// example it was categorical when the column is continuous.
    #[error(
        "Provided {ftype_req:?} datum for column {col_ix}, which is {ftype:?}"
    )]
    InvalidDatumForColumn {
        /// The column index of the offending condition
        col_ix: usize,
        /// The FType of the Datum requested
        ftype_req: FType,
        /// The actual FType of the feature at col_ix
        ftype: FType,
    },
    /// The Given is invalid
    #[error("Requested logp of 'missing' datum for column {col_ix}")]
    RequestedLogpOfMissing { col_ix: usize },
    /// One or more of the column indices in the target are out of bounds
    #[error("Target column {col_ix} invalid for state with {n_cols} columns")]
    TargetIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// One or more of the optional state indices are out of bounds
    #[error("State index {state_ix} invalid for engine with {nstates} states")]
    StateIndexOutOfBounds { nstates: usize, state_ix: usize },
    /// The user provided an empty vector for state indices rather than None
    #[error("Provided an empty states vector. Use 'None' instead")]
    NoStateIndices,
    /// The Given is invalid
    #[error("Invalid logp 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SimulateError {
    /// No targets were supplies (empty vec)
    #[error("No simulate targets provided")]
    NoTargets,
    /// One or more of the column indices in the target are out of bounds
    #[error("Target column {col_ix} invalid for state with {n_cols} columns")]
    TargetIndexOutOfBounds { n_cols: usize, col_ix: usize },
    /// One or more of the optional state indices are out of bounds
    #[error("State index {state_ix} invalid for engine with {nstates} states")]
    StateIndexOutOfBounds { nstates: usize, state_ix: usize },
    /// The user provided an empty vector for state indices rather than None
    #[error("Provided an empty states vector. Use 'None' instead")]
    NoStateIndices,
    /// The Given is invalid
    #[error("Invalid simulate 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

//! Errors that can occur in Oracle functions
use lace_cc::feature::FType;
use lace_data::Category;
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
    #[error("The column '{name}' does not exist in the table.")]
    ColumnNameDoesNotExist { name: String },
    #[error("The row '{name}' does not exist in the table.")]
    RowNameDoesNotExist { name: String },
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
    #[error("Index not found in column {col_ix} for category {cat:?}")]
    CategoryIndexNotFound { col_ix: usize, cat: Category },
}

/// Errors that can occur from bad inputs to Oracle::rowsim
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RowSimError {
    /// One of the row indices is out of bounds or
    #[error("Index error: {0}")]
    Index(#[from] IndexError),
    /// One of the column indices in wrt was out of bounds
    #[error("Invalid `wrt` index: {0}")]
    WrtColumnIndexOutOfBounds(IndexError),
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
    #[error("target index error: {0}")]
    TargetIndexOutOfBounds(IndexError),
    /// One or more of the predictor column indices is out of bounds
    #[error("predictor index error: {0}")]
    PredictorIndexOutOfBounds(IndexError),
    /// The number of QMC samples requested is zero
    #[error("Must request more than zero samples")]
    NIsZero,
}

/// Describes errors that can occur from bad inputs to
/// `Oracle::conditional_entropy`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ConditionalEntropyError {
    /// One or more of the target column indices is out of bounds
    #[error("target index error: {0}")]
    TargetIndexOutOfBounds(IndexError),
    /// One or more of the predictor column indices is out of bounds
    #[error("predictor index error: {0}")]
    PredictorIndexOutOfBounds(IndexError),
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
    #[error(
        "Requested state index {state_ix} but there are {n_states} states"
    )]
    StateIndexOutOfBounds { n_states: usize, state_ix: usize },
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

/// Describes errors that can occur from bad inputs to `Oracle::variability`
#[derive(Debug, Clone, PartialEq, Error)]
pub enum VariabilityError {
    /// The target column index is out of bounds
    #[error("Target index error in predict query: {0}")]
    IndexError(#[from] IndexError),
    /// The Given is invalid
    #[error("Invalid predict 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

/// Describes errors that arise from invalid predict uncertainty arguments
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PredictUncertaintyError {
    /// The target column index is out of bounds
    #[error("Target index error in predict uncertainty query: {0}")]
    IndexError(#[from] IndexError),
    /// The Given is invalid
    #[error("Invalid predict uncertainty 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

/// Describes errors from incompatible `col_max_logp` caches
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ColumnMaximumLogPError {
    /// The state indices used to compute the cache do not match those passed to the function.
    #[error("The state indices used to compute the cache do not match those passed to the function.")]
    InvalidStateIndices,
    /// The column indices used to compute the cache do not match those passed to the function.
    #[error("The column indices used to compute the cache do not match those passed to the function.")]
    InvalidColumnIndices,
    /// The Given conditions used to compute the cache do not match those passed to the function.
    #[error("The Given conditions used to compute the cache do not match those passed to the function.")]
    InvalidGiven,
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
    #[error("Target column index error: {0}")]
    TargetIndexOutOfBounds(IndexError),
    /// One or more of the optional state indices are out of bounds
    #[error(
        "State index {state_ix} invalid for engine with {n_states} states"
    )]
    StateIndexOutOfBounds { n_states: usize, state_ix: usize },
    /// The user provided an empty vector for state indices rather than None
    #[error("Provided an empty states vector. Use 'None' instead")]
    NoStateIndices,
    /// The Given is invalid
    #[error("Invalid logp 'given' argument: {0}")]
    GivenError(#[from] GivenError),
    #[error("Invalid `col_max_logps` argument: {0}")]
    ColumnMaximumLogPError(#[from] ColumnMaximumLogPError),
}

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SimulateError {
    /// No targets were supplies (empty vec)
    #[error("No simulate targets provided")]
    NoTargets,
    /// One or more of the column indices in the target are out of bounds
    #[error("Target column index error: {0}")]
    TargetIndexOutOfBounds(IndexError),
    /// One or more of the optional state indices are out of bounds
    #[error(
        "State index {state_ix} invalid for engine with {n_states} states"
    )]
    StateIndexOutOfBounds { n_states: usize, state_ix: usize },
    /// The user provided an empty vector for state indices rather than None
    #[error("Provided an empty states vector. Use 'None' instead")]
    NoStateIndices,
    /// The Given is invalid
    #[error("Invalid simulate 'given' argument: {0}")]
    GivenError(#[from] GivenError),
}

//! Errors that can occur in Oracle functions
use serde::Serialize;

/// Describes errors arising from a bad `Given` in the context of an Oracle
/// query.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GivenError {
    /// The `Datum` for the column at `col_ix` is the wrong type, for example it
    /// was categorical when the column is continuous.
    InvalidDatumForColumnError { col_ix: usize },
    /// The column `col_ix` appears both in the `Given` and the target
    ColumnIndexAppearsInTargetError { col_ix: usize },
    /// A column index in the given is out of bounds
    ColumnIndexOutOfBoundsError,
}

/// Describes errors that can occur from bad inputs to Oracle functions that
/// take indices are arguments
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexError {
    /// The provide row index is out of bounds
    RowIndexOutOfBoundsError,
    /// The provide column index is out of bounds
    ColumnIndexOutOfBoundsError,
}

/// Errors that can occur from bad inputs to Oracle::rowsim
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowSimError {
    /// One of the row indices is out of bounds
    RowIndexOutOfBoundsError,
    /// One or more of the column indices in `wrt` is out of bounds
    WrtColumnIndexOutOfBoundsError,
    /// The wrt was not `None`, but was an empty vector
    EmptyWrtError,
}

/// Describes errors that can occur from bad inputs to `Oracle::mi`
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MiError {
    /// Either or both of the column indices `col_a` or `col_b` is out of
    /// bounds
    ColumnIndexOutOfBoundsError,
    /// The number of QMC samples requested is zero
    NIsZeroError,
}

/// Describes errors that can occur from bad inputs to
/// `Oracle::conditional_entropy`
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntropyError {
    /// No target column indices provided
    NoTargetColumnsError,
    /// One or more of the target column indices is out of bounds
    ColumnIndexOutOfBoundsError,
    /// The number of QMC samples requested is zero
    NIsZeroError,
}

/// Describes errors that can occur from bad inputs to `Oracle::info_prop`
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InfoPropError {
    /// No target column indices provided
    NoTargetColumnsError,
    /// No predictor column indices provided
    NoPredictorColumnsError,
    /// One or more of the target column indices is out of bounds
    TargetColumnIndexOutOfBoundsError,
    /// One or more of the predictor column indices is out of bounds
    PredictorColumnIndexOutOfBoundsError,
    /// The number of QMC samples requested is zero
    NIsZeroError,
}

/// Describes errors that can occur from bad inputs to
/// `Oracle::conditional_entropy`
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConditionalEntropyError {
    /// The target column index is out of bounds
    TargetColumnIndexOutOfBoundsError,
    /// One or more The predictor column indices is out of bounds
    PredictorColumnIndexOutOfBoundsError,
    /// One or more predictor column indices occurs more than once
    DuplicatePredictorsError,
    /// No predictor columns provided
    NoPredictorColumnsError,
    /// The number of QMC samples requested is zero
    NIsZeroError,
}

/// Describes errors that can occur from bad inputs to `Oracle::surprisal`
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurprisalError {
    /// The requested row index is out of bounds
    RowIndexOutOfBoundsError,
    /// The requested column index is out of bounds
    ColumnIndexOutOfBoundsError,
    /// The `Datum` provided is incompatible with the requested column. Will
    /// not occur in `Oracle::self_surprisal`
    InvalidDatumForColumnError,
}

impl From<IndexError> for SurprisalError {
    fn from(err: IndexError) -> Self {
        match err {
            IndexError::ColumnIndexOutOfBoundsError => {
                SurprisalError::ColumnIndexOutOfBoundsError
            }
            IndexError::RowIndexOutOfBoundsError => {
                SurprisalError::RowIndexOutOfBoundsError
            }
        }
    }
}

/// Describes errors that can occur from bad inputs to `Oracle::predict`
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictError {
    /// The target column index is out of bounds
    ColumnIndexOutOfBoundsError,
    /// The Given is invalid
    GivenError(GivenError),
}

impl Into<PredictError> for GivenError {
    fn into(self) -> PredictError {
        PredictError::GivenError(self)
    }
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictUncertaintyError {
    /// The target column index is out of bounds
    ColumnIndexOutOfBoundsError,
    /// The Given is invalid
    GivenError(GivenError),
}

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogpError {
    /// No targets were supplies (empty vec)
    NoTargetsError,
    /// The number of values a row in `vals` does not equal the number of target
    /// indices in `col_ixs`
    TargetsIndicesAndValuesMismatchError,
    /// The `Datum` for the target column at `col_ix` is the wrong type, for
    /// example it was categorical when the column is continuous.
    InvalidDatumForColumnError { col_ix: usize },
    /// One or more of the column indices in the target are out of bounds
    TargetIndexOutOfBoundsError,
    /// One or more of the optional state indices are out of bounds
    StateIndexOutOfBoundsError,
    /// The number of samples requested was zero
    NIsZeroError,
    /// The Given is invalid
    GivenError(GivenError),
}

impl Into<LogpError> for GivenError {
    fn into(self) -> LogpError {
        LogpError::GivenError(self)
    }
}

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimulateError {
    /// No targets were supplies (empty vec)
    NoTargetsError,
    /// One or more of the column indices in the target are out of bounds
    TargetIndexOutOfBoundsError,
    /// One or more of the optional state indices are out of bounds
    StateIndexOutOfBoundsError,
    /// The user provided an empty vector for state indices rather than None
    NoStateIndicesError,
    /// The Given is invalid
    GivenError(GivenError),
}

impl Into<SimulateError> for GivenError {
    fn into(self) -> SimulateError {
        SimulateError::GivenError(self)
    }
}

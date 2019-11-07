/// Describes errors arising from a bad `Given` in the context of an Oracle
/// query.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum GivenError {
    /// The `Datum` for the column at `col_ix` is the wrong type, for example it
    /// was categorical when the column is continuous.
    InvalidDatumForColumnError { col_ix: usize },
    /// The column `col_ix` appears both in the `Given` and the target
    ColumnIndexAppearsInTargetError { col_ix: usize },
    /// A column index in the given is out of bounds
    ColumnIndexOutOfBoundsError,
}

impl Into<SimulateError> for GivenError {
    fn into(self) -> SimulateError {
        SimulateError::GivenError(self)
    }
}

impl Into<LogpError> for GivenError {
    fn into(self) -> LogpError {
        LogpError::GivenError(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum IndexError {
    RowIndexOutOfBoundsError,
    ColumnIndexOutOfBoundsError,
}

/// Errors that can occur from bad inputs to Oracle::rowsim
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum RowSimError {
    /// One of the row indices is out of bounds
    RowIndexOutOfBoundsError,
    /// One or more of the column indices in `wrt` is out of bounds
    WrtColumnIndexOutOfBoundsError,
    /// The wrt was not `None`, but was an empty vector
    EmptyWrtError,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum MiError {
    ColumnIndexOutOfBoundsError,
    NIsZeroError,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum EntropyError {
    NoTargetColumnsError,
    ColumnIndexOutOfBoundsError,
    NIsZeroError,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum InfoPropError {
    NoTargetColumnsError,
    NoPredictorColumnsError,
    TargetColumnIndexOutOfBoundsError,
    PredictorColumnIndexOutOfBoundsError,
    NIsZeroError,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum ConditionalEntropyError {
    TargetColumnIndexOutOfBoundsError,
    PredictorColumnIndexOutOfBoundsError,
    DuplicatePredictorsError,
    NoPredictorColumnsError,
    NIsZeroError,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum SurprisalError {
    RowIndexOutOfBoundsError,
    ColumnIndexOutOfBoundsError,
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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum PredictError {
    ColumnIndexOutOfBoundsError,
    /// The Given is invalid
    GivenError(GivenError),
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum PredictUncertaintyError {
    /// The target column index is out of bounds
    ColumnIndexOutOfBoundsError,
    /// The Given is invalid
    GivenError(GivenError),
}

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
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

/// Describes errors from bad inputs to Oracle::simulate
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum SimulateError {
    /// No targets were supplies (empty vec)
    NoTargetsError,
    /// One or more of the column indices in the target are out of bounds
    TargetIndexOutOfBoundsError,
    /// One or more of the optional state indices are out of bounds
    StateIndexOutOfBoundsError,
    /// The Given is invalid
    GivenError(GivenError),
}

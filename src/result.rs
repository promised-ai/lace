//! Result specialization to capture Braid-relevant error
use std::{error::Error as _, io, result, string::ToString};

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// The dimensions of one of more objects are not compatible for some
    /// operation
    DimensionMismatchError,
    /// Expected a column of component of one type, but received another.
    InvalidComponentTypeError,
    /// IDs for columns should be in 0, ..., N-1. This error is thrown if any
    /// are missing.
    MissingIdsError,
    /// Received an incorrect data type
    InvalidDataTypeError,
    /// The weights in a weight vector contain negative values or do not sum to
    /// 1.
    InvalidWeightsError,
    /// An algorithm took too many iterations to complete
    MaxIterationsReachedError,
    AlreadyExistsError,
    /// An assignment is invalid. Check `Assignment::validate()` for a list of
    /// all the ways an assignment can be screwed up.
    InvalidAssignmentError,
    /// Tried to reassign a datum that is already assigned. Occurs in only in
    /// the `Gibbs` row and column reassignment transitions
    AlreadyAssignedError,
    BoundsError,
    /// The input data came from an invalid source. Either tha path was not
    /// present, or the file type is not supported.
    InvalidDataSourceError,
    NotImplementedError,
    InvalidConfigError,
    IoError,
    ParseError,
    ConversionError,
}

impl ErrorKind {
    pub fn to_str(&self) -> &str {
        match self {
            ErrorKind::DimensionMismatchError => "dimension mismatch",
            ErrorKind::InvalidComponentTypeError => "invalid component type",
            ErrorKind::MissingIdsError => "missing IDs",
            ErrorKind::InvalidDataTypeError => "invalid data type",
            ErrorKind::InvalidWeightsError => "invalid weights",
            ErrorKind::MaxIterationsReachedError => {
                "maximum iterations reached"
            }
            ErrorKind::AlreadyExistsError => "already exists",
            ErrorKind::InvalidAssignmentError => "invalid assignment",
            ErrorKind::AlreadyAssignedError => "already assigned",
            ErrorKind::BoundsError => "bounds error",
            ErrorKind::InvalidDataSourceError => "invalid data source",
            ErrorKind::NotImplementedError => "not implemented",
            ErrorKind::InvalidConfigError => "invalid configuration",
            ErrorKind::IoError => "io error",
            ErrorKind::ParseError => "parse error",
            ErrorKind::ConversionError => "conversion error",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Error {
    pub kind: ErrorKind,
    pub msg: String,
}

impl ToString for Error {
    fn to_string(&self) -> String {
        format!("{}: {}", self.kind.to_str(), self.msg)
    }
}

impl Error {
    pub fn new(kind: ErrorKind, msg: String) -> Self {
        Error { kind, msg }
    }

    pub fn description(&self) -> &String {
        &self.msg
    }
}

impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        match error.into_kind() {
            csv::ErrorKind::Io(err) => {
                Error::new(ErrorKind::IoError, err.description().to_owned())
            }
            _ => Error::new(
                ErrorKind::InvalidDataSourceError,
                String::from("CSV Error"),
            ),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::new(ErrorKind::IoError, error.description().to_owned())
    }
}

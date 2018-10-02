//! Result specialization to capture Braid-relevant error
extern crate csv;

use std::error::Error as ErrorTrait;
use std::io;
use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    DimensionMismatch,
    InvalidComponentType,
    MissingIds,
    InvalidDataType,
    InvalidWeights,
    MaxIterationsReached,
    AlreadyExists,
    InvalidAssignment,
    AlreadyAssigned,
    BoundsError,
    InvalidDataSource,
    NotImplemented,
    InvalidConfig,
    IoError,
}

impl ErrorKind {
    pub fn as_str(&self) -> &str {
        match self {
            ErrorKind::DimensionMismatch => "dimension mismatch",
            ErrorKind::InvalidComponentType => "invalid component type",
            ErrorKind::MissingIds => "missing IDs",
            ErrorKind::InvalidDataType => "invalid data type",
            ErrorKind::InvalidWeights => "invalid weights",
            ErrorKind::MaxIterationsReached => "maximum iterations reached",
            ErrorKind::AlreadyExists => "already exists",
            ErrorKind::InvalidAssignment => "invalid assignment",
            ErrorKind::AlreadyAssigned => "already assigned",
            ErrorKind::BoundsError => "bounds error",
            ErrorKind::InvalidDataSource => "invalid data source",
            ErrorKind::NotImplemented => "not implemented",
            ErrorKind::InvalidConfig => "invalid configuration",
            ErrorKind::IoError => "io error",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Error {
    kind: ErrorKind,
    msg: String,
}

impl Error {
    pub fn new(kind: ErrorKind, msg: &str) -> Self {
        Error {
            kind,
            msg: String::from(msg),
        }
    }

    pub fn description(&self) -> &str {
        self.msg.as_str()
    }
}

impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        match error.into_kind() {
            csv::ErrorKind::Io(e) => {
                Error::new(ErrorKind::IoError, e.description())
            }
            _ => Error::new(ErrorKind::InvalidDataSource, "CSV Error"),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::new(ErrorKind::IoError, error.description())
    }
}

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

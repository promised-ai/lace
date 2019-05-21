use crate::result;
use serde::{Deserialize, Serialize};
use std::convert::{From, TryFrom};

/// A type of data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(rename = "datum")]
pub enum Datum {
    #[serde(rename = "continuous")]
    Continuous(f64),
    #[serde(rename = "categorical")]
    Categorical(u8),
    #[serde(rename = "binary")]
    Binary(bool),
    #[serde(rename = "missing")]
    Missing, // Should carry an error message?
}

impl TryFrom<Datum> for f64 {
    type Error = result::Error;

    fn try_from(datum: Datum) -> Result<f64, Self::Error> {
        match datum {
            Datum::Continuous(x) => Ok(x),
            _ => {
                let kind = result::ErrorKind::ConversionError;
                let msg = String::from("Can only convert Continuous into f64");
                Err(result::Error::new(kind, msg))
            }
        }
    }
}

impl TryFrom<Datum> for u8 {
    type Error = result::Error;

    fn try_from(datum: Datum) -> Result<u8, Self::Error> {
        match datum {
            Datum::Categorical(x) => Ok(x),
            _ => {
                let kind = result::ErrorKind::ConversionError;
                let msg = String::from("Can only convert Categorical into u8");
                Err(result::Error::new(kind, msg))
            }
        }
    }
}

impl TryFrom<Datum> for bool {
    type Error = result::Error;

    fn try_from(datum: Datum) -> Result<bool, Self::Error> {
        match datum {
            Datum::Binary(x) => Ok(x),
            _ => {
                let kind = result::ErrorKind::ConversionError;
                let msg = String::from("Can only convert Binary into bool");
                Err(result::Error::new(kind, msg))
            }
        }
    }
}

impl From<&Datum> for String {
    fn from(datum: &Datum) -> String {
        match datum {
            Datum::Continuous(x) => format!("{}", *x),
            Datum::Categorical(x) => format!("{}", *x),
            Datum::Binary(x) => format!("{}", *x),
            Datum::Missing => String::from("NaN"),
        }
    }
}

// XXX: What happens when we add vector types? Error?
impl Datum {
    /// Unwraps the datum as an `f64` if possible
    pub fn to_f64_opt(&self) -> Option<f64> {
        match self {
            Datum::Continuous(x) => Some(*x),
            Datum::Categorical(x) => Some(*x as f64),
            Datum::Binary(x) => {
                if *x {
                    Some(1.0)
                } else {
                    Some(0.0)
                }
            }
            Datum::Missing => None,
        }
    }

    /// Unwraps the datum as an `u8` if possible
    pub fn to_u8_opt(&self) -> Option<u8> {
        match self {
            Datum::Continuous(..) => None,
            Datum::Categorical(x) => Some(*x),
            Datum::Binary(x) => {
                if *x {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            Datum::Missing => None,
        }
    }

    /// Returns `true` if the `Datum` is continuous
    pub fn is_continuous(&self) -> bool {
        match self {
            Datum::Continuous(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the `Datum` is categorical
    pub fn is_categorical(&self) -> bool {
        match self {
            Datum::Categorical(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the `Datum` is binary
    pub fn is_binary(&self) -> bool {
        match self {
            Datum::Binary(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the `Datum` is missing
    pub fn is_missing(&self) -> bool {
        match self {
            Datum::Missing => true,
            _ => false,
        }
    }
}

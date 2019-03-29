extern crate serde;

use serde::{Deserialize, Serialize};

/// A type of data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

// XXX: What happens when we add vector types? Error?
impl Datum {
    /// Unwraps the datum as an `f64` if possible
    pub fn as_f64(&self) -> Option<f64> {
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
    pub fn as_u8(&self) -> Option<u8> {
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

    /// Returns the datum as a string
    pub fn as_string(&self) -> String {
        match self {
            Datum::Continuous(x) => format!("{}", *x),
            Datum::Categorical(x) => format!("{}", *x),
            Datum::Binary(x) => format!("{}", *x),
            Datum::Missing => String::from("NaN"),
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

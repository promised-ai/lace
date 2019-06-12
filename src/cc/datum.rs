use std::convert::{From, TryFrom};

use braid_stats::labeler::Label;
use serde::{Deserialize, Serialize};

use crate::result;

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
    #[serde(rename = "label")]
    Label(Label),
    #[serde(rename = "missing")]
    Missing, // Should carry an error message?
}

macro_rules! impl_try_from_datum {
    ($out: ty, $pat_in: path, $msg: expr) => {
        impl TryFrom<Datum> for $out {
            type Error = result::Error;

            fn try_from(datum: Datum) -> Result<$out, Self::Error> {
                match datum {
                    $pat_in(x) => Ok(x),
                    _ => {
                        let kind = result::ErrorKind::ConversionError;
                        let msg = String::from(
                            "Can only convert Continuous into f64",
                        );
                        Err(result::Error::new(kind, msg))
                    }
                }
            }
        }
    };
}

impl_try_from_datum!(
    f64,
    Datum::Continuous,
    "Can only convert Continuous to f64"
);
impl_try_from_datum!(
    u8,
    Datum::Categorical,
    "Can only convert Categorical to u8"
);
impl_try_from_datum!(bool, Datum::Binary, "Can only convert Binary to bool");
impl_try_from_datum!(Label, Datum::Label, "Can only convert Label");

impl From<&Datum> for String {
    fn from(datum: &Datum) -> String {
        match datum {
            Datum::Continuous(x) => format!("{}", *x),
            Datum::Categorical(x) => format!("{}", *x),
            Datum::Binary(x) => format!("{}", *x),
            Datum::Label(x) => {
                let truth_str = match x.truth {
                    Some(y) => {
                        if y {
                            "1"
                        } else {
                            "0"
                        }
                    }
                    None => "None",
                };
                let label_str = if x.label { "1" } else { "0" };
                format!("IL({}, {})", label_str, truth_str)
            }
            Datum::Missing => String::from("NaN"),
        }
    }
}

// XXX: What happens when we add vector types? Error?
impl Datum {
    /// Unwraps the datum as an `f64` if possible. The conversion will coerce
    /// from other types if possible.
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::Datum;
    /// assert_eq!(Datum::Continuous(1.2).to_f64_opt(), Some(1.2));
    /// assert_eq!(Datum::Categorical(8).to_f64_opt(), Some(8.0));
    /// assert_eq!(Datum::Binary(true).to_f64_opt(), Some(1.0));
    /// assert_eq!(Datum::Binary(false).to_f64_opt(), Some(0.0));
    /// assert_eq!(Datum::Missing.to_f64_opt(), None);
    /// ```
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
            Datum::Label(..) => None,
        }
    }

    /// Unwraps the datum as an `u8` if possible. The conversion will coerce
    /// from other types if possible.
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::Datum;
    /// assert_eq!(Datum::Continuous(1.2).to_u8_opt(), None);
    /// assert_eq!(Datum::Categorical(8).to_u8_opt(), Some(8));
    /// assert_eq!(Datum::Binary(true).to_u8_opt(), Some(1));
    /// assert_eq!(Datum::Binary(false).to_u8_opt(), Some(0));
    /// assert_eq!(Datum::Missing.to_u8_opt(), None);
    /// ```
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
            Datum::Label(..) => None,
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

    /// Returns `true` if the `Datum` is label
    pub fn is_label(&self) -> bool {
        match self {
            Datum::Label(_) => true,
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

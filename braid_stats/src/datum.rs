use crate::labeler::Label;
use serde::{Deserialize, Serialize};
use std::convert::{From, TryFrom};
use thiserror::Error;

/// Represents the types of data braid can work with
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(rename = "datum")]
pub enum Datum {
    #[serde(rename = "continuous")]
    Continuous(f64),
    #[serde(rename = "categorical")]
    Categorical(u8),
    #[serde(rename = "label")]
    Label(Label),
    #[serde(rename = "missing")]
    Missing,
}

/// Describes an error converting from a Datum to another type
#[derive(Debug, Clone, Error, PartialEq)]
pub enum DatumConversionError {
    /// Tried to convert Continuous into a type other than f64
    #[error("tried to convert Continuous into a type other than f64")]
    InvalidTypeRequestedFromContinuous,
    /// Tried to convert Categorical into a type other than u8
    #[error("tried to convert Categorical into a type other than u8")]
    InvalidTypeRequestedFromCategorical,
    /// Tried to convert Label into a type other than Label
    #[error("tried to convert Label into a type other than Label")]
    InvalidTypeRequestedFromLabel,
    /// Cannot convert Missing into a value of any type
    #[error("cannot convert Missing into a value of any type")]
    CannotConvertMissing,
}

macro_rules! impl_try_from_datum {
    ($out: ty, $pat_in: path, $err: expr) => {
        impl TryFrom<Datum> for $out {
            type Error = DatumConversionError;

            fn try_from(datum: Datum) -> Result<$out, Self::Error> {
                match datum {
                    $pat_in(x) => Ok(x),
                    Datum::Missing => {
                        Err(DatumConversionError::CannotConvertMissing)
                    }
                    _ => Err($err),
                }
            }
        }
    };
}

impl_try_from_datum!(
    f64,
    Datum::Continuous,
    DatumConversionError::InvalidTypeRequestedFromContinuous
);

impl_try_from_datum!(
    u8,
    Datum::Categorical,
    DatumConversionError::InvalidTypeRequestedFromCategorical
);

impl_try_from_datum!(
    Label,
    Datum::Label,
    DatumConversionError::InvalidTypeRequestedFromLabel
);

impl From<&Datum> for String {
    fn from(datum: &Datum) -> String {
        match datum {
            Datum::Continuous(x) => format!("{}", *x),
            Datum::Categorical(x) => format!("{}", *x),
            Datum::Label(x) => {
                let truth_str = match x.truth {
                    Some(y) => y.to_string(),
                    None => String::from("None"),
                };
                let label_str = x.label.to_string();
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
    /// # use braid_stats::Datum;
    /// assert_eq!(Datum::Continuous(1.2).to_f64_opt(), Some(1.2));
    /// assert_eq!(Datum::Categorical(8).to_f64_opt(), Some(8.0));
    /// assert_eq!(Datum::Missing.to_f64_opt(), None);
    /// ```
    pub fn to_f64_opt(&self) -> Option<f64> {
        match self {
            Datum::Continuous(x) => Some(*x),
            Datum::Categorical(x) => Some(f64::from(*x)),
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
    /// # use braid_stats::Datum;
    /// assert_eq!(Datum::Continuous(1.2).to_u8_opt(), None);
    /// assert_eq!(Datum::Categorical(8).to_u8_opt(), Some(8));
    /// assert_eq!(Datum::Missing.to_u8_opt(), None);
    /// ```
    pub fn to_u8_opt(&self) -> Option<u8> {
        match self {
            Datum::Continuous(..) => None,
            Datum::Categorical(x) => Some(*x),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn continuous_datum_try_into_f64() {
        let datum = Datum::Continuous(1.1);
        let _res: f64 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn continuous_datum_try_into_u8_panics() {
        let datum = Datum::Continuous(1.1);
        let _res: u8 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn continuous_datum_try_into_label_panics() {
        let datum = Datum::Continuous(1.1);
        let _res: Label = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn missing_datum_try_into_u8_panics() {
        let datum = Datum::Missing;
        let _res: u8 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn missing_datum_try_into_f64_panics() {
        let datum = Datum::Missing;
        let _res: f64 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn missing_datum_try_into_label_panics() {
        let datum = Datum::Missing;
        let _res: Label = datum.try_into().unwrap();
    }

    #[test]
    fn categorical_datum_try_into_u8() {
        let datum = Datum::Categorical(7);
        let _res: u8 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn categorical_datum_try_into_f64_panics() {
        let datum = Datum::Categorical(7);
        let _res: f64 = datum.try_into().unwrap();
    }
}

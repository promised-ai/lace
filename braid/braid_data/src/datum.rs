use crate::label::Label;
use serde::{Deserialize, Serialize};
use std::convert::{From, TryFrom};
use std::hash::Hash;
use thiserror::Error;

/// Represents the types of data lace can work with
#[derive(Debug, Clone, Serialize, Deserialize, PartialOrd)]
#[serde(rename = "datum")]
pub enum Datum {
    #[serde(rename = "binary")]
    Binary(bool),
    #[serde(rename = "continuous")]
    Continuous(f64),
    #[serde(rename = "categorical")]
    Categorical(u8),
    #[serde(rename = "label")]
    Label(Label),
    #[serde(rename = "count")]
    Count(u32),
    #[serde(rename = "missing")]
    Missing,
}

/// Describes an error converting from a Datum to another type
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DatumConversionError {
    /// Tried to convert Binary into a type other than bool
    #[error("tried to convert Binary into a type other than bool")]
    InvalidTypeRequestedFromBinary,
    /// Tried to convert Continuous into a type other than f64
    #[error("tried to convert Continuous into a type other than f64")]
    InvalidTypeRequestedFromContinuous,
    /// Tried to convert Categorical into a type other than u8
    #[error("tried to convert Categorical into a type other than u8")]
    InvalidTypeRequestedFromCategorical,
    /// Tried to convert Label into a type other than Label
    #[error("tried to convert Label into a type other than Label")]
    InvalidTypeRequestedFromLabel,
    /// Tried to convert Count into a type other than u32
    #[error("tried to convert Count into a type other than u32")]
    InvalidTypeRequestedFromCount,
    /// Cannot convert Missing into a value of any type
    #[error("cannot convert Missing into a value of any type")]
    CannotConvertMissing,
}

fn hash_float<H: std::hash::Hasher>(float: f64, state: &mut H) {
    // Note that IEEE 754 doesn’t define just a single NaN value
    let x: f64 = if float.is_nan() { std::f64::NAN } else { float };

    x.to_bits().hash(state);
}

impl Hash for Datum {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Binary(x) => x.hash(state),
            Self::Continuous(x) => hash_float(*x, state),
            Self::Categorical(x) => x.hash(state),
            Self::Label(x) => x.hash(state),
            Self::Count(x) => x.hash(state),
            Self::Missing => hash_float(std::f64::NAN, state),
        }
    }
}

macro_rules! datum_peq {
    ($x: ident, $y: ident, $variant: ident) => {{
        if let Datum::$variant(y) = $y {
            $x == y
        } else {
            false
        }
    }};
}

// PartialEq and Hash must agree with each other.
impl PartialEq for Datum {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Continuous(x) => {
                if let Self::Continuous(y) = other {
                    if x.is_nan() && y.is_nan() {
                        true
                    } else {
                        x == y
                    }
                } else {
                    false
                }
            }
            Self::Binary(x) => datum_peq!(x, other, Binary),
            Self::Categorical(x) => datum_peq!(x, other, Categorical),
            Self::Label(x) => datum_peq!(x, other, Label),
            Self::Count(x) => datum_peq!(x, other, Count),
            Self::Missing => matches!(other, Self::Missing),
        }
    }
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
    bool,
    Datum::Binary,
    DatumConversionError::InvalidTypeRequestedFromBinary
);

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
    u32,
    Datum::Count,
    DatumConversionError::InvalidTypeRequestedFromCount
);

impl_try_from_datum!(
    Label,
    Datum::Label,
    DatumConversionError::InvalidTypeRequestedFromLabel
);

impl From<&Datum> for String {
    fn from(datum: &Datum) -> String {
        match datum {
            Datum::Binary(x) => format!("{}", *x),
            Datum::Continuous(x) => format!("{}", *x),
            Datum::Categorical(x) => format!("{}", *x),
            Datum::Count(x) => format!("{}", *x),
            Datum::Label(x) => {
                let truth_str = match x.truth {
                    Some(y) => y.to_string(),
                    None => String::from("None"),
                };
                let label_str = x.label.to_string();
                format!("IL({label_str}, {truth_str})")
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
    /// # use lace_data::Datum;
    /// assert_eq!(Datum::Continuous(1.2).to_f64_opt(), Some(1.2));
    /// assert_eq!(Datum::Categorical(8).to_f64_opt(), Some(8.0));
    /// assert_eq!(Datum::Missing.to_f64_opt(), None);
    /// ```
    pub fn to_f64_opt(&self) -> Option<f64> {
        match self {
            Datum::Binary(x) => Some(if *x { 1.0 } else { 0.0 }),
            Datum::Continuous(x) => Some(*x),
            Datum::Categorical(x) => Some(f64::from(*x)),
            Datum::Count(x) => Some(f64::from(*x)),
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
    /// # use lace_data::Datum;
    /// assert_eq!(Datum::Continuous(1.2).to_u8_opt(), None);
    /// assert_eq!(Datum::Categorical(8).to_u8_opt(), Some(8));
    /// assert_eq!(Datum::Missing.to_u8_opt(), None);
    /// ```
    pub fn to_u8_opt(&self) -> Option<u8> {
        match self {
            Datum::Binary(..) => None,
            Datum::Continuous(..) => None,
            Datum::Categorical(x) => Some(*x),
            Datum::Count(..) => None,
            Datum::Missing => None,
            Datum::Label(..) => None,
        }
    }

    /// Returns `true` if the `Datum` is binary
    pub fn is_binary(&self) -> bool {
        matches!(self, Datum::Binary(_))
    }

    /// Returns `true` if the `Datum` is continuous
    pub fn is_continuous(&self) -> bool {
        matches!(self, Datum::Continuous(_))
    }

    /// Returns `true` if the `Datum` is categorical
    pub fn is_categorical(&self) -> bool {
        matches!(self, Datum::Categorical(_))
    }

    /// Returns `true` if the `Datum` is label
    pub fn is_label(&self) -> bool {
        matches!(self, Datum::Label(_))
    }

    /// Returns `true` if the `Datum` is Count
    pub fn is_count(&self) -> bool {
        matches!(self, Datum::Count(_))
    }

    /// Returns `true` if the `Datum` is missing
    pub fn is_missing(&self) -> bool {
        matches!(self, Datum::Missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    // TODO macro this away into something like this:
    // try_into_tests (
    //  { Continuous: { passes: [f64], fails [u8, u32, Label] }},
    //  { Categorical: { passes: [u8, u32, f64], fails [Label] }},
    //  { Missing: { passes: [], fails [f64, u8, u32, Label] }},
    //  { Label: { passes: [Label], fails [f64, u8, u32] }},
    //  { Count: { passes: [u32], fails [f64, u8, u32, Label] }},
    // );
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

    #[test]
    fn count_data_into_f64() {
        let datum = Datum::Count(12);
        let x = datum.to_f64_opt();
        assert_eq!(x, Some(12.0));
    }

    #[test]
    fn count_data_try_into_u32() {
        let datum = Datum::Count(12);
        let _x: u32 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn count_data_try_into_u8_fails() {
        let datum = Datum::Count(12);
        let _x: u8 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn count_data_try_into_label_fails() {
        let datum = Datum::Count(12);
        let _x: Label = datum.try_into().unwrap();
    }

    #[test]
    fn serde_continuous() {
        let data = r#"
            {
                "continuous": 1.2
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Continuous(1.2));
    }

    #[test]
    fn serde_categorical() {
        let data = r#"
            {
                "categorical": 2
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Categorical(2));
    }

    #[test]
    fn serde_count() {
        let data = r#"
            {
                "count": 277
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Count(277));
    }

    #[test]
    fn serde_label() {
        let data = r#"
            {
                "label": {
                   "label": 2,
                   "truth": 3
                }
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Label(Label::new(2, Some(3))));
    }

    #[test]
    fn serde_missing() {
        let data = r#""missing""#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Missing);
    }
}

use crate::Category;
use serde::{Deserialize, Serialize};
use std::convert::{From, TryFrom};
use std::hash::Hash;
use thiserror::Error;

/// Represents the types of data lace can work with
#[derive(Debug, Clone, Serialize, Deserialize, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum Datum {
    Binary(bool),
    Continuous(f64),
    Categorical(Category),
    Count(u32),
    Missing,
    #[cfg(feature = "experimental")]
    Index(usize),
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
    /// Tried to convert Categorical into non-categorical type
    #[error("tried to convert Categorical into non-categorical type")]
    InvalidTypeRequestedFromCategorical,
    /// Tried to convert Count into a type other than u32
    #[error("tried to convert Count into a type other than u32")]
    InvalidTypeRequestedFromCount,
    /// Cannot convert Missing into a value of any type
    #[error("cannot convert Missing into a value of any type")]
    CannotConvertMissing,
    /// Tried to convert Index into a type other than usize
    #[cfg(feature = "experimental")]
    #[error("tried to convert Index into a type other than usize")]
    InvalidTypeRequestedFromIndex,
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
            Self::Count(x) => x.hash(state),
            Self::Missing => hash_float(std::f64::NAN, state),
            #[cfg(feature = "experimental")]
            Self::Index(x) => x.hash(state),
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
            Self::Count(x) => datum_peq!(x, other, Count),
            Self::Missing => matches!(other, Self::Missing),
            #[cfg(feature = "experimental")]
            Self::Index(x) => datum_peq!(x, other, Index),
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

impl TryFrom<Datum> for u8 {
    type Error = DatumConversionError;

    fn try_from(datum: Datum) -> Result<u8, Self::Error> {
        match datum {
            Datum::Categorical(Category::U8(x)) => Ok(x),
            Datum::Categorical(Category::Bool(x)) => Ok(x as u8),
            #[cfg(feature = "experimental")]
            Datum::Missing => Err(DatumConversionError::CannotConvertMissing),
            _ => Err(DatumConversionError::InvalidTypeRequestedFromCategorical),
        }
    }
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
    u32,
    Datum::Count,
    DatumConversionError::InvalidTypeRequestedFromCount
);

#[cfg(feature = "experimental")]
impl_try_from_datum!(
    usize,
    Datum::Index,
    DatumConversionError::InvalidTypeRequestedFromIndex
);

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
    /// assert_eq!(Datum::Categorical(true.into()).to_f64_opt(), Some(1.0));
    /// assert_eq!(Datum::Categorical(8_u8.into()).to_f64_opt(), Some(8.0));
    /// assert_eq!(Datum::Categorical("cat".into()).to_f64_opt(), None);
    /// assert_eq!(Datum::Missing.to_f64_opt(), None);
    /// ```
    pub fn to_f64_opt(&self) -> Option<f64> {
        match self {
            Datum::Binary(x) => Some(if *x { 1.0 } else { 0.0 }),
            Datum::Continuous(x) => Some(*x),
            Datum::Categorical(Category::Bool(x)) => {
                Some(if *x { 1.0 } else { 0.0 })
            }
            Datum::Categorical(Category::U8(x)) => Some(f64::from(*x)),
            Datum::Categorical(Category::String(_)) => None,
            Datum::Count(x) => Some(f64::from(*x)),
            Datum::Missing => None,
            #[cfg(feature = "experimental")]
            Datum::Index(x) => None,
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
    /// assert_eq!(Datum::Categorical(8_u8.into()).to_u8_opt(), Some(8));
    /// assert_eq!(Datum::Categorical(true.into()).to_u8_opt(), Some(1));
    /// assert_eq!(Datum::Categorical("cat".into()).to_u8_opt(), None);
    /// assert_eq!(Datum::Missing.to_u8_opt(), None);
    /// ```
    pub fn to_u8_opt(&self) -> Option<u8> {
        match self {
            Datum::Binary(..) => None,
            Datum::Continuous(..) => None,
            Datum::Categorical(Category::U8(x)) => Some(*x),
            Datum::Categorical(Category::Bool(x)) => Some(*x as u8),
            Datum::Categorical(Category::String(_)) => None,
            Datum::Count(..) => None,
            Datum::Missing => None,
            #[cfg(feature = "experimental")]
            Datum::Index(x) => None,
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

    /// Returns `true` if the `Datum` is Count
    pub fn is_count(&self) -> bool {
        matches!(self, Datum::Count(_))
    }

    /// Returns `true` if the `Datum` is missing
    pub fn is_missing(&self) -> bool {
        matches!(self, Datum::Missing)
    }

    /// Returns `true` if the `Datum` is Index type
    #[cfg(feature = "experimental")]
    pub fn is_index(&self) -> bool {
        matches!(self, Datum::Index(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    // TODO macro this away
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
    fn categorical_datum_try_into_u8() {
        let datum = Datum::Categorical(Category::U8(7));
        let _res: u8 = datum.try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn categorical_datum_try_into_f64_panics() {
        let datum = Datum::Categorical(Category::U8(7));
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
    fn serde_continuous() {
        let data = r#"
            {
                "continuous": 1.2
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Continuous(1.2));
    }

    #[test]
    fn serde_categorical_u8() {
        let data = r#"
            {
                "categorical": 2
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Categorical(Category::U8(2)));
    }

    #[test]
    fn serde_categorical_bool() {
        let data = r#"
            {
                "categorical": true
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Categorical(Category::Bool(true)));
    }

    #[test]
    fn serde_categorical_string() {
        let data = r#"
            {
                "categorical": "zoidberg"
            }"#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Categorical("zoidberg".into()));
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
    fn serde_missing() {
        let data = r#""missing""#;

        let x: Datum = serde_json::from_str(data).unwrap();

        assert_eq!(x, Datum::Missing);
    }
}

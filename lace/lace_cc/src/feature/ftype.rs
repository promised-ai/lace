use std::convert::TryFrom;

use lace_codebook::ColType;
use lace_data::Datum;
use serde::{Deserialize, Serialize};

/// Feature type
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FType {
    Binary,
    Continuous,
    Categorical,
    Count,
}

impl std::fmt::Display for FType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binary => write!(f, "Binary"),
            Self::Continuous => write!(f, "Continuous"),
            Self::Categorical => write!(f, "Categorical"),
            Self::Count => write!(f, "Count"),
        }
    }
}

impl std::str::FromStr for FType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Binary" => Ok(FType::Binary),
            "Continuous" => Ok(FType::Continuous),
            "Categorical" => Ok(FType::Categorical),
            "Count" => Ok(FType::Count),
            invalid => Err(format!("Invalid ftype: '{invalid}'")),
        }
    }
}

impl From<FType> for String {
    fn from(ftype: FType) -> Self {
        ftype.to_string()
    }
}

/// FType compatibility information
pub struct FTypeCompat {
    /// The FType of the Datum passed to this feature
    pub ftype_req: FType,
    /// The FType of this feature
    pub ftype: FType,
}

impl TryFrom<&Datum> for FType {
    type Error = ();

    fn try_from(datum: &Datum) -> Result<Self, Self::Error> {
        match datum {
            Datum::Binary(_) => Ok(FType::Binary),
            Datum::Categorical(_) => Ok(FType::Categorical),
            Datum::Continuous(_) => Ok(FType::Continuous),
            Datum::Count(_) => Ok(FType::Count),
            Datum::Missing => Err(()),
        }
    }
}

impl FType {
    pub fn from_coltype(coltype: &ColType) -> FType {
        match coltype {
            ColType::Continuous { .. } => FType::Continuous,
            ColType::Categorical { .. } => FType::Categorical,
            ColType::Count { .. } => FType::Count,
        }
    }

    pub fn is_continuous(self) -> bool {
        matches!(self, FType::Continuous)
    }

    pub fn is_categorical(self) -> bool {
        matches!(self, FType::Categorical)
    }

    pub fn is_count(self) -> bool {
        matches!(self, FType::Count)
    }

    /// Return a tuple
    pub fn datum_compatible(self, datum: &Datum) -> (bool, FTypeCompat) {
        if let Ok(ftype_req) = FType::try_from(datum) {
            let ok = ftype_req == self;
            (
                ok,
                FTypeCompat {
                    ftype_req,
                    ftype: self,
                },
            )
        } else {
            // always compatible if the datum is a missing value
            (
                true,
                FTypeCompat {
                    ftype_req: self,
                    ftype: self,
                },
            )
        }
    }
}

// TODO: tests

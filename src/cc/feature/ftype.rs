use braid_codebook::ColType;
use braid_stats::Datum;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

/// Feature type
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FType {
    #[serde(rename = "continuous")]
    Continuous,
    #[serde(rename = "categorical")]
    Categorical,
    #[serde(rename = "labeler")]
    Labeler,
    #[serde(rename = "count")]
    Count,
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
            Datum::Categorical(_) => Ok(FType::Categorical),
            Datum::Continuous(_) => Ok(FType::Continuous),
            Datum::Label(_) => Ok(FType::Labeler),
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
            ColType::Labeler { .. } => FType::Labeler,
        }
    }

    pub fn is_continuous(self) -> bool {
        match self {
            FType::Continuous => true,
            _ => false,
        }
    }

    pub fn is_categorical(self) -> bool {
        match self {
            FType::Categorical => true,
            _ => false,
        }
    }

    pub fn is_labeler(self) -> bool {
        match self {
            FType::Labeler => true,
            _ => false,
        }
    }

    pub fn is_count(self) -> bool {
        match self {
            FType::Count => true,
            _ => false,
        }
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SummaryStatistics {
    #[serde(rename = "continuous")]
    Continuous {
        min: f64,
        max: f64,
        mean: f64,
        median: f64,
        variance: f64,
    },
    #[serde(rename = "categorical")]
    Categorical { min: u8, max: u8, mode: Vec<u8> },
    #[serde(rename = "labeler")]
    Labeler {
        n: usize,
        n_true: usize,
        n_false: usize,
        n_labeled: usize,
        n_correct: usize,
    },
    #[serde(rename = "count")]
    Count {
        min: u32,
        max: u32,
        median: f64,
        mean: f64,
        mode: Vec<u32>,
    },
}

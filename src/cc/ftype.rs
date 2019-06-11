use serde::{Deserialize, Serialize};

/// Feature type
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FType {
    #[serde(rename = "continuous")]
    Continuous,
    #[serde(rename = "categorical")]
    Categorical,
    #[serde(rename = "labeler")]
    Labeler,
}

impl FType {
    pub fn is_continuous(&self) -> bool {
        match self {
            FType::Continuous => true,
            _ => false,
        }
    }

    pub fn is_categorical(&self) -> bool {
        match self {
            FType::Categorical => true,
            _ => false,
        }
    }

    pub fn is_labeler(&self) -> bool {
        match self {
            FType::Labeler => true,
            _ => false,
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
}

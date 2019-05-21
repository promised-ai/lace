extern crate serde;

use serde::{Deserialize, Serialize};

/// Feature type
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FType {
    #[serde(rename = "continuous")]
    Continuous,
    #[serde(rename = "categorical")]
    Categorical,
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
}

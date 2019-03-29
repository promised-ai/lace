extern crate serde;

use serde::{Deserialize, Serialize};

/// Feature type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FType {
    #[serde(rename = "continuous")]
    Continuous,
    #[serde(rename = "categorical")]
    Categorical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

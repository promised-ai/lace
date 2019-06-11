use crate::cc::DataContainer;
use braid_stats::labeler::Label;
use serde::{Deserialize, Serialize};

/// Used when pulling data from features for saving
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum FeatureData {
    /// Univariate continuous data
    Continuous(DataContainer<f64>),
    /// Categorical data
    Categorical(DataContainer<u8>),
    /// Categorical data
    Labeler(DataContainer<Label>),
}

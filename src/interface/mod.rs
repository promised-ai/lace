//! User-interface objects for running and querying
mod engine;
mod given;
mod metadata;
mod oracle;

pub use engine::BuildEngineError;
pub use engine::Engine;
pub use engine::EngineBuilder;
pub use engine::EngineSaver;
pub use engine::{
    AppendStrategy, InsertDataActions, InsertMode, OverwriteMode, Row,
    SupportExtension, Value, WriteMode,
};
pub use metadata::Metadata;
pub use oracle::utils;

pub use oracle::{
    ConditionalEntropyType, DatalessOracle, ImputeUncertaintyType,
    MiComponents, MiType, Oracle, OracleT, PredictUncertaintyType,
};

pub use given::Given;

pub mod error {
    pub use super::engine::error::*;
    pub use super::given::IntoGivenError;
    pub use super::oracle::error::*;
}

use crate::cc::State;
use crate::cc::SummaryStatistics;
use braid_stats::Datum;

/// Returns references to crosscat states
pub trait HasStates {
    fn states(&self) -> &Vec<State>;
    fn states_mut(&mut self) -> &mut Vec<State>;
}

/// Returns and summrize data
pub trait HasData {
    /// Summarize the data in a feature
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics;
    /// Return the datum in a cell
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum;
}

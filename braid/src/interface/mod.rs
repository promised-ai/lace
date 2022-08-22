//! User-interface objects for running and querying
mod engine;
mod given;
mod metadata;
mod oracle;

pub use braid_metadata::latest::Metadata;
pub use engine::{
    create_comms, AppendStrategy, BuildEngineError, Builder, Engine,
    InsertDataActions, InsertMode, OverwriteMode, Row, StateProgress,
    StateProgressMonitor, SupportExtension, Value, WriteMode,
};
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

use braid_cc::state::State;
use braid_data::{Datum, SummaryStatistics};

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

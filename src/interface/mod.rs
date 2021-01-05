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

use crate::cc::{Feature, State};
use braid_stats::Datum;

/// Returns references to crosscat states
pub trait HasStates {
    fn states(&self) -> &Vec<State>;
    fn states_mut(&mut self) -> &mut Vec<State>;
}

impl HasStates for Oracle {
    #[inline]
    fn states(&self) -> &Vec<State> {
        &self.states
    }

    #[inline]
    fn states_mut(&mut self) -> &mut Vec<State> {
        &mut self.states
    }
}

impl HasStates for Engine {
    #[inline]
    fn states(&self) -> &Vec<State> {
        &self.states
    }

    #[inline]
    fn states_mut(&mut self) -> &mut Vec<State> {
        &mut self.states
    }
}

use crate::cc::SummaryStatistics;

/// Returns and summrize data
pub trait HasData {
    /// Summarize the data in a feature
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics;
    /// Return the datum in a cell
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum;
}

impl HasData for Oracle {
    #[inline]
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics {
        self.data.0[&ix].summarize()
    }

    #[inline]
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum {
        self.data.get(row_ix, col_ix)
    }
}

impl HasData for Engine {
    #[inline]
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics {
        let state = &self.states[0];
        let view_ix = state.asgn.asgn[ix];
        // XXX: Cloning the data could be very slow
        state.views[view_ix].ftrs[&ix].clone_data().summarize()
    }

    #[inline]
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum {
        self.states[0].datum(row_ix, col_ix)
    }
}

impl OracleT for Oracle {}
impl OracleT for Engine {}

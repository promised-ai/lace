//! User-interface objects for running and querying
mod engine;
mod given;
mod oracle;

pub use engine::Engine;
pub use engine::EngineBuilder;
pub use engine::EngineSaver;
pub use engine::RowAlignmentStrategy;
pub use oracle::utils;

pub use oracle::{
    ConditionalEntropyType, ImputeUncertaintyType, MiComponents, MiType,
    Oracle, OracleT, PredictUncertaintyType,
};

pub use given::Given;

pub mod error {
    pub use super::engine::error::*;
    pub use super::given::IntoGivenError;
    pub use super::oracle::error::*;
}

use crate::cc::{FeatureData, State};
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

/// Returns data
pub trait HasData {
    fn column(&self, ix: usize) -> &FeatureData;
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum;
}

impl HasData for Oracle {
    #[inline]
    fn column(&self, ix: usize) -> &FeatureData {
        &self.data.0[&ix]
    }

    #[inline]
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum {
        self.data.get(row_ix, col_ix)
    }
}

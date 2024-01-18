//! User-interface objects for running and querying
mod engine;
mod given;
mod metadata;
mod oracle;

pub use engine::{
    update_handler, AppendStrategy, BuildEngineError, Engine, EngineBuilder,
    InsertDataActions, InsertMode, OverwriteMode, Row, SupportExtension, Value,
    WriteMode,
};
use lace_codebook::Codebook;
pub use lace_metadata::latest::Metadata;
pub use oracle::utils;

pub use oracle::{
    ConditionalEntropyType, DatalessOracle, MiComponents, MiType, Oracle,
    OracleT, RowSimilarityVariant,
};

pub use given::Given;

pub mod error {
    pub use super::engine::error::*;
    pub use super::given::IntoGivenError;
    pub use super::oracle::error::*;
}

use lace_cc::state::State;
use lace_data::{Datum, SummaryStatistics};

/// Returns references to crosscat states
pub trait HasStates {
    /// Get a reference to the States
    fn states(&self) -> &Vec<State>;

    /// Get a mutable reference to the States
    fn states_mut(&mut self) -> &mut Vec<State>;

    /// Get the number of states
    fn n_states(&self) -> usize {
        self.states().len()
    }

    /// Get the number of rows in the states
    fn n_rows(&self) -> usize {
        self.states()[0].n_rows()
    }

    /// Get the number of columns in the states
    fn n_cols(&self) -> usize {
        self.states()[0].n_cols()
    }
}

/// Returns and summarizes data
pub trait HasData {
    /// Summarize the data in a feature
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics;
    /// Return the datum in a cell
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum;
}

/// Returns a codebook
pub trait HasCodebook {
    fn codebook(&self) -> &Codebook;
}

pub trait CanOracle: HasStates + HasData + HasCodebook + Sync {}

impl<T: HasStates + HasData + HasCodebook + Sync> CanOracle for T {}

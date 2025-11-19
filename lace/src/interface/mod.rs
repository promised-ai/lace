//! User-interface objects for running and querying
mod engine;
mod given;
mod metadata;
mod oracle;

pub use engine::update_handler;
pub use engine::AppendStrategy;
pub use engine::BuildEngineError;
pub use engine::Engine;
pub use engine::EngineBuilder;
pub use engine::InsertDataActions;
pub use engine::InsertMode;
pub use engine::OverwriteMode;
pub use engine::Row;
pub use engine::SupportExtension;
pub use engine::Value;
pub use engine::WriteMode;
pub use given::Given;
pub use oracle::ConditionalEntropyType;
pub use oracle::DatalessOracle;
pub use oracle::MiComponents;
pub use oracle::MiType;
pub use oracle::Oracle;
pub use oracle::OracleT;
pub use oracle::RowSimilarityVariant;
pub use oracle::Variability;

use crate::codebook::Codebook;
pub use crate::metadata::latest::Metadata;

pub mod error {
    pub use super::engine::error::*;
    pub use super::given::IntoGivenError;
    pub use super::oracle::error::*;
}

use crate::cc::state::State;
use crate::data::Datum;
use crate::data::SummaryStatistics;

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

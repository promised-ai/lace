//! Cross-categorization components
pub mod alg;
pub mod assignment;
pub mod component;
pub mod config;
pub mod container;
pub mod data_store;
pub mod datum;
mod feature;
pub mod file_utils;
pub mod state;
pub mod state_builder;
pub mod transition;
pub mod view;

pub use alg::{ColAssignAlg, RowAssignAlg};
pub use assignment::{Assignment, AssignmentBuilder};
pub use component::ConjugateComponent;
pub use container::DataContainer;
pub use data_store::DataStore;
pub use datum::Datum;
pub use state::State;
pub use state_builder::StateBuilder;
pub use transition::StateTransition;
pub use view::{View, ViewBuilder};

pub use feature::{
    geweke, ColModel, Column, Component, FType, Feature, FeatureData,
    SummaryStatistics,
};

pub struct AppendRowsData {
    pub col_ix: usize,
    pub data: Vec<Datum>,
}

impl AppendRowsData {
    pub fn new(col_ix: usize, data: Vec<Datum>) -> Self {
        AppendRowsData { col_ix, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

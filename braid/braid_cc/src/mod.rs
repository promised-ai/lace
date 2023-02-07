//! Cross-categorization components
mod alg;
pub mod assignment;
pub mod component;
pub mod config;
pub mod data_store;
mod feature;
pub mod file_utils;
pub mod state;
pub mod transition;
pub mod view;

pub use alg::{ColAssignAlg, RowAssignAlg};
pub use assignment::{Assignment, AssignmentBuilder};
pub use component::ConjugateComponent;
pub use config::{EngineUpdateConfig, StateUpdateConfig};
pub use data_store::DataStore;
pub use state::State;
pub use transition::StateTransition;
pub use view::{View, ViewBuilder};

use lace_data::Datum;

pub use feature::{
    geweke, ColModel, Column, Component, FType, Feature, FeatureData,
    SummaryStatistics, TranslateDatum,
};

pub(crate) use feature::FeatureHelper;

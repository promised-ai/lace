pub mod alg;
pub mod assignment;
pub mod column_model;
pub mod component;
pub mod config;
pub mod container;
pub mod data_store;
pub mod datum;
pub mod feature;
pub mod file_utils;
pub mod ftype;
pub mod state;
pub mod transition;
pub mod view;

pub use crate::cc::alg::{ColAssignAlg, RowAssignAlg};

pub use crate::cc::assignment::{Assignment, AssignmentBuilder};
pub use crate::cc::column_model::ColModel;
pub use crate::cc::component::ConjugateComponent;
pub use crate::cc::container::{DataContainer, FeatureData};
pub use crate::cc::data_store::DataStore;
pub use crate::cc::datum::Datum;
pub use crate::cc::feature::{Column, Feature};
pub use crate::cc::ftype::FType;
pub use crate::cc::state::State;
pub use crate::cc::view::{View, ViewBuilder};

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

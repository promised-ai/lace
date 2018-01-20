pub mod assignment;
pub mod column_model;
pub mod container;
pub mod feature;
pub mod view;
pub mod state;
pub mod codebook;

pub use cc::assignment::Assignment;
pub use cc::column_model::{ColModel, ColModelType, DType};
pub use cc::feature::Feature;
pub use cc::feature::Column;
pub use cc::view::View;
pub use cc::container::DataContainer;
pub use cc::state::State;
pub use cc::codebook::Codebook;


#[derive(Serialize, Deserialize)]
pub struct StatesAndCodebook {
    pub states: Vec<State>,
    pub codebook: Codebook
}

impl StatesAndCodebook {
    pub fn new(states: Vec<State>, codebook: Codebook) -> Self {
        StatesAndCodebook { states: states, codebook: codebook }
    }
}

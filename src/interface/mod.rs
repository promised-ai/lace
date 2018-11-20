pub mod bencher;
pub mod engine;
pub mod engine_builder;
pub mod oracle;
pub mod utils;

use cc::Datum;

pub use interface::bencher::Bencher;
pub use interface::engine::Engine;
pub use interface::engine_builder::EngineBuilder;
pub use interface::oracle::MiType;
pub use interface::oracle::Oracle;

/// Describes a the conditions (or not) on a conditional distribution
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Given {
    /// The conditions in `(column_id, value)` tuples. The tuple
    /// `(12, Datum::Continuous(2.3))` indicates that we wish to condition on
    /// the value of column 12 being 2.3.
    Conditions(Vec<(usize, Datum)>),
    /// The absence of conditioning observations
    Nothing,
}

impl Given {
    pub fn is_nothing(&self) -> bool {
        match self {
            Given::Nothing => true,
            _ => false,
        }
    }

    pub fn is_conditions(&self) -> bool {
        match self {
            Given::Conditions(..) => true,
            _ => false,
        }
    }
}

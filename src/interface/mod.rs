//! User-interface objects for running and querying
mod bencher;
mod engine;
mod oracle;

use braid_stats::Datum;
use serde::{Deserialize, Serialize};

pub use engine::Engine;
pub use engine::EngineBuilder;
pub use engine::EngineSaver;

pub use oracle::ConditionalEntropyType;
pub use oracle::ImputeUncertaintyType;
pub use oracle::MiComponents;
pub use oracle::MiType;
pub use oracle::Oracle;
pub use oracle::PredictUncertaintyType;

pub use bencher::Bencher;
pub use bencher::BencherResult;
pub use bencher::BencherRig;

/// Describes a the conditions (or not) on a conditional distribution
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, PartialOrd)]
pub enum Given {
    /// The conditions in `(column_id, value)` tuples. The tuple
    /// `(12, Datum::Continuous(2.3))` indicates that we wish to condition on
    /// the value of column 12 being 2.3.
    #[serde(rename = "conditions")]
    Conditions(Vec<(usize, Datum)>),
    /// The absence of conditioning observations
    #[serde(rename = "nothing")]
    Nothing,
}

impl Given {
    /// Determine whether there are no conditions
    ///
    /// # Example
    ///
    /// ```
    /// # use braid_stats::Datum;
    /// # use braid::Given;
    /// let nothing_given = Given::Nothing;
    ///
    /// assert!(nothing_given.is_nothing());
    ///
    /// let something_given = Given::Conditions(vec![(0, Datum::Categorical(1))]);
    ///
    /// assert!(!something_given.is_nothing());
    /// ```
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

impl Default for Given {
    fn default() -> Self {
        Given::Nothing
    }
}

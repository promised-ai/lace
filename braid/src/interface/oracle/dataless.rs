use braid_cc::state::State;
use braid_codebook::Codebook;
use braid_data::{Datum, SummaryStatistics};
use braid_metadata::latest::Metadata;
use serde::{Deserialize, Serialize};

use crate::{interface::HasCodebook, HasData, HasStates, Oracle};

/// An oracle without data for sensitive data applications
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, from = "Metadata", into = "Metadata")]
pub struct DatalessOracle {
    /// Vector of states
    pub states: Vec<State>,
    /// Metadata for the rows and columns
    pub codebook: Codebook,
}

impl From<Oracle> for DatalessOracle {
    fn from(oracle: Oracle) -> Self {
        Self {
            states: oracle.states,
            codebook: oracle.codebook,
        }
    }
}

impl HasStates for DatalessOracle {
    #[inline]
    fn states(&self) -> &Vec<State> {
        &self.states
    }

    #[inline]
    fn states_mut(&mut self) -> &mut Vec<State> {
        &mut self.states
    }
}

impl HasData for DatalessOracle {
    #[inline]
    fn summarize_feature(&self, _ix: usize) -> SummaryStatistics {
        SummaryStatistics::None
    }

    #[inline]
    fn cell(&self, _row_ix: usize, _col_ix: usize) -> Datum {
        Datum::Missing
    }
}

impl HasCodebook for DatalessOracle {
    fn codebook(&self) -> &Codebook {
        &self.codebook
    }
}

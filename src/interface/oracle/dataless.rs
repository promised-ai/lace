use braid_codebook::Codebook;
use braid_stats::Datum;
use serde::{Deserialize, Serialize};

use crate::cc::{State, SummaryStatistics};
use crate::{HasData, HasStates, Oracle, OracleT};
use braid_metadata::latest::Metadata;

/// An oracle without data for sensitive data applications
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, from = "Metadata", into = "Metadata")]
pub struct DatalessOracle {
    /// Vector of states
    pub states: Vec<State>,
    /// Metadata for the rows and columns
    pub codebook: Codebook,
}

impl OracleT for DatalessOracle {}

impl From<Oracle> for DatalessOracle {
    fn from(oracle: Oracle) -> DatalessOracle {
        DatalessOracle {
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

//! User-interface objects for running and querying
mod engine;
mod given;
mod metadata;
mod oracle;

pub use braid_metadata::latest::Metadata;
pub use engine::BuildEngineError;
pub use engine::Engine;
pub use engine::EngineBuilder;
pub use engine::EngineSaver;
pub use engine::{
    AppendStrategy, InsertDataActions, InsertMode, OverwriteMode, Row,
    SupportExtension, Value, WriteMode,
};
pub use oracle::utils;

pub use oracle::{
    ConditionalEntropyType, DatalessOracle, ImputeUncertaintyType,
    MiComponents, MiType, Oracle, OracleT, PredictUncertaintyType,
};

pub use given::Given;

pub mod error {
    pub use super::engine::error::*;
    pub use super::given::IntoGivenError;
    pub use super::oracle::error::*;
}

use crate::cc::State;
use crate::cc::SummaryStatistics;
use braid_codebook::Codebook;
use braid_stats::Datum;

/// Returns references to crosscat states
pub trait HasStates {
    fn states(&self) -> &Vec<State>;
    fn states_mut(&mut self) -> &mut Vec<State>;
}

/// Returns and summrize data
pub trait HasData {
    /// Summarize the data in a feature
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics;
    /// Return the datum in a cell
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum;
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NameOrIndex {
    Name(String),
    Index(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Index {
    Row(NameOrIndex),
    Column(NameOrIndex),
    Cell(NameOrIndex, NameOrIndex),
}

impl Index {
    #[inline]
    pub fn is_row(&self) -> bool {
        match self {
            &Index::Row(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_column(&self) -> bool {
        match self {
            &Index::Column(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_cell(&self) -> bool {
        match self {
            &Index::Cell(_, _) => true,
            _ => false,
        }
    }

    /// Convert an index to an integer type index.
    #[inline]
    pub fn to_usize_index(self, codebook: &Codebook) -> Option<Self> {
        match self {
            Self::Row(NameOrIndex::Index(_)) => Some(self),
            Self::Column(NameOrIndex::Index(_)) => Some(self),
            Self::Cell(NameOrIndex::Index(_), NameOrIndex::Index(_)) => {
                Some(self)
            }
            Self::Row(NameOrIndex::Name(name)) => codebook
                .row_index(name.as_str())
                .map(|ix| Index::Row(NameOrIndex::Index(ix))),
            Self::Column(NameOrIndex::Name(name)) => codebook
                .column_index(name.as_str())
                .map(|ix| Index::Column(NameOrIndex::Index(ix))),
            Self::Cell(NameOrIndex::Name(row), NameOrIndex::Name(col)) => {
                codebook
                    .row_index(row.as_str())
                    .map(|ix| NameOrIndex::Index(ix))
                    .and_then(|rowix| {
                        codebook.column_index(col.as_str()).map(|ix| {
                            let colix = NameOrIndex::Index(ix);
                            Index::Cell(rowix, colix)
                        })
                    })
            }
            Self::Cell(NameOrIndex::Name(row), colix) => {
                codebook.row_index(row.as_str()).map(|ix| {
                    let rowix = NameOrIndex::Index(ix);
                    Index::Cell(rowix, colix)
                })
            }
            Self::Cell(rowix, NameOrIndex::Name(col)) => {
                codebook.column_index(col.as_str()).map(|ix| {
                    let colix = NameOrIndex::Index(ix);
                    Index::Cell(rowix, colix)
                })
            }
        }
    }
}

impl From<usize> for NameOrIndex {
    fn from(ix: usize) -> NameOrIndex {
        NameOrIndex::Index(ix)
    }
}

impl From<String> for NameOrIndex {
    fn from(name: String) -> NameOrIndex {
        NameOrIndex::Name(name)
    }
}

impl<T1, T2> From<(T1, T2)> for Index
where
    T1: Into<NameOrIndex>,
    T2: Into<NameOrIndex>,
{
    fn from(ixs: (T1, T2)) -> Index {
        Index::Cell(ixs.0.into(), ixs.1.into())
    }
}

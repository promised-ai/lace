//! A fast, extensible probabilistic cross-categorization engine.
//!
//! # Example
//!
//! Analyze structure in the pre-run `animals` example. Each row is an animal
//! and each column is a feature of that animal. The feature is present if the
//! cell value is 1 and is absent if the value is 0.
//!
//! First, we create an oracle and import some `enum`s that allow us to call
//! out some of the row and column indices in plain English.
//!
//! ```rust
//! use lace::examples::Example;
//! use lace::OracleT;
//!
//! let oracle = Example::Animals.oracle().unwrap();
//! ```
//!  Let's ask about the statistical dependence between whether something swims
//! and is fast or has flippers. We expect the something swimming is more
//! indicative of whether it swims than whether something is fast, therefore we
//! expect the dependence between swims and flippers to be higher.
//!
//! ```rust
//! # use lace::examples::Example;
//! # use lace::examples::animals::{Row, Column};
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use lace::OracleT;
//! let depprob_fast = oracle.depprob(
//!     "swims",
//!     "fast",
//! ).unwrap();
//!
//! let depprob_flippers = oracle.depprob(
//!     "swims",
//!     "flippers",
//! ).unwrap();
//!
//! assert!(depprob_flippers > depprob_fast);
//! ```
//!
//! We have the same expectation of mutual information. Mutual information
//! requires more input from the user. We need to know what type of mutual
//! information, how many samples to take if we need to estimate the mutual
//! information, and a random number generator for the Monte Carlo integrator.
//!
//! ```rust
//! # use lace::examples::Example;
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use lace::OracleT;
//! use lace::MiType;
//!
//! let mut rng = rand::thread_rng();
//!
//! let mi_fast = oracle.mi(
//!     "swims",
//!     "fast",
//!     1000,
//!     MiType::Iqr,
//! ).unwrap();
//!
//! let mi_flippers = oracle.mi(
//!     "swims",
//!     "flippers",
//!     1000,
//!     MiType::Iqr,
//! ).unwrap();
//!
//! assert!(mi_flippers > mi_fast);
//! ```
//!
//! We can likewise ask about the similarity between rows -- in this case,
//! animals.
//!
//! ```
//! # use lace::examples::Example;
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use lace::OracleT;
//! use lace::RowSimilarityVariant;
//!
//! let wrt: Option<&[usize]> = None;
//! let rowsim_wolf = oracle.rowsim(
//!     "wolf",
//!     "chihuahua",
//!     wrt,
//!     RowSimilarityVariant::ViewWeighted,
//! ).unwrap();
//!
//! let rowsim_rat = oracle.rowsim(
//!     "rat",
//!     "chihuahua",
//!     wrt,
//!     RowSimilarityVariant::ViewWeighted,
//! ).unwrap();
//!
//! assert!(rowsim_rat > rowsim_wolf);
//! ```
//!
//! And we can add context to similarity.
//!
//! ```
//! # use lace::examples::Example;
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use lace::OracleT;
//! # use lace::RowSimilarityVariant;
//! let context = vec!["swims"];
//! let rowsim_otter = oracle.rowsim(
//!     "beaver",
//!     "otter",
//!     Some(&context),
//!     RowSimilarityVariant::ViewWeighted,
//! ).unwrap();
//!
//! let rowsim_dolphin = oracle.rowsim(
//!     "beaver",
//!     "dolphin",
//!     Some(&context),
//!     RowSimilarityVariant::ViewWeighted,
//! ).unwrap();
//! ```
#![warn(unused_extern_crates)]
#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone,
    clippy::perf
)]

pub mod bencher;
pub mod config;
pub mod data;
pub mod defaults;
pub mod examples;
mod interface;
pub mod misc;
pub mod optimize;

mod index;

pub use index::*;

pub use config::EngineUpdateConfig;

pub use interface::{
    utils, utils::ColumnMaximumLogpCache, AppendStrategy, BuildEngineError,
    Builder, ConditionalEntropyType, DatalessOracle, Engine, Given, HasData,
    HasStates, ImputeUncertaintyType, InsertDataActions, InsertMode, Metadata,
    MiComponents, MiType, Oracle, OracleT, OverwriteMode,
    PredictUncertaintyType, Row, RowSimilarityVariant, StateProgress,
    StateProgressMonitor, SupportExtension, Value, WriteMode,
};

pub mod error {
    pub use super::interface::error::*;
}

use serde::Serialize;
use std::fmt::Debug;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ParseError<T: Serialize + Debug + Clone + PartialEq + Eq>(T);

impl<T> std::string::ToString for ParseError<T>
where
    T: Serialize + Debug + Clone + PartialEq + Eq,
{
    fn to_string(&self) -> String {
        format!("{self:?}")
    }
}

pub use lace_cc::feature::FType;
pub use lace_cc::state::StateDiagnostics;
pub use lace_cc::transition::StateTransition;
pub use lace_data::{Datum, SummaryStatistics};

pub mod consts {
    pub use lace_consts::*;
}

pub mod metadata {
    pub use lace_metadata::*;
}

pub mod codebook {
    pub use lace_codebook::*;
}

pub mod cc {
    pub use lace_cc::*;
}

pub mod stats {
    pub use lace_stats::*;
}

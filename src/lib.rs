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
//! use braid::examples::Example;
//! use braid::examples::animals::{Row, Column};
//! use braid::OracleT;
//!
//! let oracle = Example::Animals.oracle().unwrap();
//! ```
//!  Let's ask about the statistical dependence between whether something swims
//! and is fast or has flippers. We expect the something swimming is more
//! indicative of whether it swims than whether something is fast, therefore we
//! expect the dependence between swims and flippers to be higher.
//!
//! ```rust
//! # use braid::examples::Example;
//! # use braid::examples::animals::{Row, Column};
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use braid::OracleT;
//! let depprob_fast = oracle.depprob(
//!     Column::Swims.into(),
//!     Column::Fast.into(),
//! ).unwrap();
//!
//! let depprob_flippers = oracle.depprob(
//!     Column::Swims.into(),
//!     Column::Flippers.into(),
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
//! # use braid::examples::Example;
//! # use braid::examples::animals::{Row, Column};
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use braid::OracleT;
//! use braid::MiType;
//!
//! let mut rng = rand::thread_rng();
//!
//! let mi_fast = oracle.mi(
//!     Column::Swims.into(),
//!     Column::Fast.into(),
//!     1000,
//!     MiType::Iqr,
//! ).unwrap();
//!
//! let mi_flippers = oracle.mi(
//!     Column::Swims.into(),
//!     Column::Flippers.into(),
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
//! # use braid::examples::Example;
//! # use braid::examples::animals::Row;
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use braid::OracleT;
//! let rowsim_wolf = oracle.rowsim(
//!     Row::Wolf.into(),
//!     Row::Chihuahua.into(),
//!     None
//! ).unwrap();
//!
//! let rowsim_rat = oracle.rowsim(
//!     Row::Rat.into(),
//!     Row::Chihuahua.into(),
//!     None
//! ).unwrap();
//!
//! assert!(rowsim_rat > rowsim_wolf);
//! ```
//!
//! And we can add context to similarity.
//!
//! ```
//! # use braid::examples::Example;
//! # use braid::examples::animals::{Column, Row};
//! # let oracle = Example::Animals.oracle().unwrap();
//! # use braid::OracleT;
//! let context = vec![Column::Swims.into()];
//! let rowsim_otter = oracle.rowsim(
//!     Row::Beaver.into(),
//!     Row::Otter.into(),
//!     Some(&context),
//! ).unwrap();
//!
//! let rowsim_dolphin = oracle.rowsim(
//!     Row::Beaver.into(),
//!     Row::Dolphin.into(),
//!     Some(&context),
//! ).unwrap();
//! ```
pub mod benchmark;
pub mod cc;
pub mod data;
pub mod defaults;
pub mod dist;
pub mod examples;
pub mod file_config;
mod interface;
pub mod misc;
pub mod optimize;
pub mod testers;

pub use interface::{
    utils, ConditionalEntropyType, Engine, EngineBuilder, EngineSaver, Given,
    ImputeUncertaintyType, InsertMode, InsertOverwrite, MiComponents, MiType,
    Oracle, OracleT, PredictUncertaintyType, Row, RowAlignmentStrategy, Value,
    HasData, HasStates
};

pub mod error {
    pub use super::interface::error::*;
}

use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParseError<T: Serialize + Debug + Clone + PartialEq + Eq + Hash>(T);

impl<T> std::string::ToString for ParseError<T>
where
    T: Serialize + Debug + Clone + PartialEq + Eq + Hash,
{
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

//! A fast, extensible probabilistic cross-categorization engine.
pub mod cc;
pub mod data;
pub mod defaults;
pub mod dist;
pub mod examples;
pub mod interface;
pub mod misc;
pub mod optimize;
pub mod result;
pub mod testers;

pub use crate::interface::{Engine, EngineBuilder, Oracle};

pub use crate::cc::Datum;
pub use crate::interface::Given;
pub use crate::result::{Error, ErrorKind, Result};

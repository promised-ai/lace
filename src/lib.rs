// #![feature(try_from)]
extern crate num_cpus;
extern crate rayon;

pub mod cc;
pub mod data;
pub mod dist;
pub mod enumeration;
pub mod geweke;
pub mod interface;
pub mod misc;
pub mod optimize;
pub mod result;

pub use crate::cc::Codebook;
pub use crate::interface::Engine;
pub use crate::interface::EngineBuilder;
pub use crate::interface::Oracle;

pub use crate::cc::Datum;
pub use crate::interface::Given;
pub use crate::result::{Error, ErrorKind, Result};

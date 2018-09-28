#![feature(extern_prelude)]
#![feature(try_from)]

#[macro_use]
extern crate approx;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate log;
#[macro_use]
extern crate maplit;

extern crate num_cpus;
extern crate rayon;

pub mod cc;
pub mod data;
pub mod defaults;
pub mod dist;
pub mod enumeration;
pub mod geweke;
pub mod interface;
pub mod misc;
pub mod optimize;
mod result;
pub mod stats;

pub use cc::Codebook;
pub use interface::Engine;
pub use interface::EngineBuilder;
pub use interface::Oracle;

pub use result::{Error, ErrorKind, Result};

#![feature(extern_prelude)]
#![feature(try_from)]

#[macro_use]
extern crate approx;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate log;

extern crate num_cpus;
extern crate rayon;

pub mod cc;
pub mod data;
pub mod dist;
pub mod geweke;
pub mod interface;
pub mod misc;
pub mod optimize;
pub mod special;
pub mod stats;

pub use cc::Codebook;
pub use interface::Engine;
pub use interface::EngineBuilder;
pub use interface::Oracle;

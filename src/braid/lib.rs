#![feature(try_from)]
#![feature(match_default_bindings)]

#[macro_use]
extern crate approx;
#[macro_use]
extern crate serde_derive;

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

pub use interface::Engine;
pub use interface::Oracle;

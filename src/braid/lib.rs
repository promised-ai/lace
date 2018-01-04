#![feature(try_from)]
#![feature(match_default_bindings)]
#[macro_use] extern crate approx;
#[macro_use] extern crate serde_derive;

extern crate rayon;

pub mod dist;
pub mod misc;
pub mod special;
pub mod optimize;
pub mod cc;
pub mod geweke;
pub mod data;
pub mod interface;

pub use interface::Oracle;
pub use interface::Engine;

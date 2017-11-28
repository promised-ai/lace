#![feature(type_ascription)]
#[macro_use] extern crate approx;
#[macro_use] extern crate serde_derive;
extern crate rayon;

pub mod dist;
pub mod misc;
pub mod special;
pub mod cc;
pub mod geweke;

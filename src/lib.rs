// #![feature(try_from)]
pub mod cc;
pub mod data;
pub mod defaults;
pub mod dist;
pub mod enumeration;
pub mod geweke;
pub mod integrate;
pub mod interface;
pub mod misc;
pub mod optimize;
pub mod result;

pub use crate::interface::{Engine, EngineBuilder, Oracle};

pub use crate::{
    cc::Datum,
    interface::Given,
    result::{Error, ErrorKind, Result},
};

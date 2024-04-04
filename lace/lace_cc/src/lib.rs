#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]

pub mod alg;
pub mod component;
pub mod config;
pub mod constrain;
pub mod feature;
pub mod massflip;
pub mod state;
pub mod traits;
pub mod transition;
pub mod view;

use serde::Serialize;
use std::fmt::{Debug, Display};

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ParseError<T: Serialize + Debug + Clone + PartialEq + Eq>(T);

impl<T> Display for ParseError<T>
where
    T: Serialize + Debug + Clone + PartialEq + Eq,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<T> std::error::Error for ParseError<T> where
    T: Serialize + Debug + Clone + PartialEq + Eq
{
}

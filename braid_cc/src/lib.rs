pub mod alg;
pub mod assignment;
pub mod component;
pub mod config;
pub mod feature;
pub mod misc;
pub mod state;
pub mod traits;
pub mod transition;
pub mod view;

use serde::Serialize;
use std::fmt::Debug;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ParseError<T: Serialize + Debug + Clone + PartialEq + Eq>(T);

impl<T> std::string::ToString for ParseError<T>
where
    T: Serialize + Debug + Clone + PartialEq + Eq,
{
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

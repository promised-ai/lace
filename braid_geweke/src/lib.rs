//! Geweke (joint distribution) test
mod tester;
mod traits;

pub use tester::{GewekeResult, GewekeTester};
pub use traits::{GewekeModel, GewekeResampleData, GewekeSummarize};

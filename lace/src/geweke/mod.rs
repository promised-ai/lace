//! Geweke (joint distribution) test
#![warn(unused_extern_crates)]
#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]

mod tester;
mod traits;

pub use tester::GewekeResult;
pub use tester::GewekeTester;
pub use traits::GewekeModel;
pub use traits::GewekeResampleData;
pub use traits::GewekeSummarize;

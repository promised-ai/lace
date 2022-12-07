#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]

pub mod api;
pub mod result;
pub mod server;
pub mod utils;
mod version;

pub use version::CRATE_VERSION;

#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]
//! Default values for priors and inference-type things
use rv::dist::Gamma;

/// The default number of iterations of the mh_prior sampler
pub const MH_PRIOR_ITERS: usize = 50;

/// Default alpha prior for Geweke
pub fn geweke_alpha_prior() -> Gamma {
    Gamma::new(3.0, 3.0).unwrap()
}

/// Default alpha prior in general
pub fn general_alpha_prior() -> Gamma {
    Gamma::new(1.0, 1.0).unwrap()
}

/// Default alpha prior for State assignment of columns to views
pub fn state_alpha_prior() -> Gamma {
    Gamma::new(1.0, 1.0).unwrap()
}

/// Default alpha prior for View assignment of rows to categories
pub fn view_alpha_prior() -> Gamma {
    Gamma::new(1.0, 1.0).unwrap()
}

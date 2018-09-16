pub mod cdf;
pub mod chi_square;
pub mod ks;
pub mod mh;
pub mod mixture_type;
pub mod perm;
pub mod pit;

pub use stats::cdf::EmpiricalCdf;
pub use stats::chi_square::chi_square_test;
pub use stats::ks::{ks2sample, ks_test};
pub use stats::mixture_type::MixtureType;
pub use stats::perm::perm_test;
pub use stats::pit::pit;

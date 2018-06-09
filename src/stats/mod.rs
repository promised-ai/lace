pub mod chi_square;
pub mod ks;
pub mod perm;
pub mod cdf;

pub use stats::chi_square::chi_square_test;
pub use stats::ks::{ks2sample, ks_test};
pub use stats::perm::perm_test;
pub use stats::cdf::EmpiricalCdf;

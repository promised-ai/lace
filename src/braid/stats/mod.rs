pub mod chi_square;
pub mod ks;

pub use stats::chi_square::chi_square_test;
pub use stats::ks::{ks_test, ks2sample};

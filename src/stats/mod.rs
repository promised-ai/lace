pub mod chi_square;
pub mod ks;
pub mod perm;

pub use stats::chi_square::chi_square_test;
pub use stats::ks::{ks_test, ks2sample};
pub use stats::perm::perm_test;

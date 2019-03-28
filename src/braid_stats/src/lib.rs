pub mod cdf;
pub mod chi_square;
pub mod defaults;
pub mod ks;
pub mod mh;
pub mod mixture_type;
pub mod perm;
pub mod prior;

mod sample_error;

pub use cdf::EmpiricalCdf;
pub use chi_square::chi_square_test;
pub use ks::{ks2sample, ks_test};
pub use mixture_type::MixtureType;
pub use perm::perm_test;
pub use sample_error::SampleError;

extern crate rand;
extern crate rv;

use rand::Rng;
use rv::traits::Rv;

pub trait UpdatePrior<X, Fx: Rv<X>> {
    /// Draw new prior parameters given a set of existing models and the hyper
    /// prior.
    fn update_prior<R: Rng>(&mut self, components: &Vec<&Fx>, rng: &mut R);
}

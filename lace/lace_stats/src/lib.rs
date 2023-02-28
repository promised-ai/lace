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
mod cdf;
mod chi_square;
pub mod dist;
mod entropy;
pub mod integrate;
mod ks;
pub mod mat;
pub mod mh;
mod mixture_type;
mod perm;
pub mod prior;
pub mod seq;
mod simplex;

mod sample_error;

pub use cdf::EmpiricalCdf;
pub use entropy::QmcEntropy;
pub use lace_consts::rv;
pub use mixture_type::MixtureType;
pub use perm::L2Norm;
pub use sample_error::SampleError;
pub use simplex::*;

use itertools::iproduct;
use rand::Rng;
use rv::traits::{KlDivergence, Rv};

pub mod test {
    use super::{chi_square, ks, perm};

    pub use chi_square::chi_square_test;
    pub use ks::{ks2sample, ks_test};
    pub use perm::{gauss_kernel, gauss_perm_test, perm_test};
}

/// Trait defining methods to update prior hyperparameters
pub trait UpdatePrior<X, Fx: Rv<X>, H> {
    /// Draw new prior parameters given a set of existing models and the hyper
    /// prior. Returns the likelihood of the components given the new prior
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Fx],
        hyper: &H,
        rng: &mut R,
    ) -> f64;
}

impl UpdatePrior<bool, crate::rv::dist::Bernoulli, ()>
    for crate::rv::dist::Beta
{
    fn update_prior<R: rand::Rng>(
        &mut self,
        _components: &[&crate::rv::dist::Bernoulli],
        _hyper: &(),
        _rng: &mut R,
    ) -> f64 {
        0.0
    }
}

/// Compute the Jensen-Shannon divergence of all Components of a Mixture
pub trait MixtureJsd {
    fn mixture_jsd(&self) -> f64;
}

/// Compute pairwise KL divergence on collections of probability distributions
pub trait PairwiseKl {
    /// Sum of pairwise KL Divergences
    fn pairwise_kl(&self) -> f64;
}

impl<Fx> PairwiseKl for Vec<Fx>
where
    Fx: KlDivergence,
{
    fn pairwise_kl(&self) -> f64 {
        iproduct!(self, self).fold(0_f64, |acc, (f1, f2)| acc + f1.kl(f2))
    }
}

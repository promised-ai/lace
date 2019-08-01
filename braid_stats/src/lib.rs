#![feature(euclidean_division)]

pub mod cdf;
pub mod chi_square;
pub mod integrate;
pub mod ks;
pub mod labeler;
pub mod mh;
mod mixture_type;
pub mod perm;
pub mod prior;
pub mod seq;
pub mod simplex;

mod sample_error;

pub use cdf::EmpiricalCdf;
pub use chi_square::chi_square_test;
pub use ks::{ks2sample, ks_test};
pub use mixture_type::MixtureType;
pub use perm::perm_test;
pub use sample_error::SampleError;

use itertools::iproduct;
use rand::Rng;
use rv::traits::{KlDivergence, Rv};

/// Trait defining methods to update prior hyperparameters
pub trait UpdatePrior<X, Fx: Rv<X>> {
    /// Draw new prior parameters given a set of existing models and the hyper
    /// prior.
    fn update_prior<R: Rng>(&mut self, components: &Vec<&Fx>, rng: &mut R);
}

/// Compute the Jensen-shannon divergence of all comonents of a Mixture
pub trait MixtureJsd {
    fn mixture_jsd(&self) -> f64;
}

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

mod cdf;
mod chi_square;
mod datum;
mod entropy;
pub mod integrate;
mod ks;
pub mod labeler;
pub mod mh;
mod mixture_type;
mod perm;
pub mod prior;
pub mod seq;
mod simplex;

mod sample_error;

pub use cdf::EmpiricalCdf;
pub use datum::{Datum, DatumConversionError};
pub use entropy::QmcEntropy;
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
pub trait UpdatePrior<X, Fx: Rv<X>> {
    /// Draw new prior parameters given a set of existing models and the hyper
    /// prior. Returns the likelihood of the components given the new prior
    fn update_prior<R: Rng>(&mut self, components: &[&Fx], rng: &mut R) -> f64;
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

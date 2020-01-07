use crate::Datum;
use std::vec::Drain;

/// Trait supporting the estimation of join entropies from Quasi Monte Carlo
/// sequences.
pub trait QmcEntropy {
    /// Return the number of dimensions in a QMC sequence is required to
    /// generate a `Datum`
    fn us_needed(&self) -> usize;
    /// Take `ndims` uniformly sample f64 in (0, 1) and convert them into a
    /// `Datum`.
    fn us_to_datum(&self, us: &mut Drain<f64>) -> Datum;
    /// The reciprocal of the importance PDF/PMF.
    fn q_recip(&self) -> f64;
}

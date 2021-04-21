//! Probability distribution utilities and traits
pub mod stick_breaking;
pub mod traits;

use std::convert::TryFrom;
use std::fmt::Debug;

use braid_stats::{Datum, UpdatePrior};
use rv::traits::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::cc::Component;
use crate::dist::traits::AccumScore;

const HALF_LN_PI: f64 = 0.572_364_942_924_700_1;
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7;

/// A Braid-ready datum.
pub trait BraidDatum:
    Sync + Serialize + DeserializeOwned + TryFrom<Datum> + Default + Clone + Debug
{
}

impl<X> BraidDatum for X where
    X: Sync
        + Serialize
        + DeserializeOwned
        + TryFrom<Datum>
        + Default
        + Clone
        + Debug
{
}

/// A Braid-ready datum.
pub trait BraidStat:
    Sync + Serialize + DeserializeOwned + Debug + Clone + PartialEq
{
}
impl<X> BraidStat for X where
    X: Sync + Serialize + DeserializeOwned + Debug + Clone + PartialEq
{
}

/// A Braid-ready likelihood function, f(x).
pub trait BraidLikelihood<X: BraidDatum>:
    Rv<X>
    + Mode<X>
    + AccumScore<X>
    + HasSuffStat<X>
    + Serialize
    + DeserializeOwned
    + Sync
    + Into<Component>
    + Clone
    + Debug
    + PartialEq
{
}

impl<X, Fx> BraidLikelihood<X> for Fx
where
    X: BraidDatum,
    Fx: Rv<X>
        + Mode<X>
        + AccumScore<X>
        + HasSuffStat<X>
        + Serialize
        + DeserializeOwned
        + Sync
        + Into<Component>
        + Clone
        + Debug
        + PartialEq,
    Fx::Stat: Sync + Serialize + DeserializeOwned + Clone + Debug,
{
}

/// A Braid-ready prior π(f)
pub trait BraidPrior<X: BraidDatum, Fx: BraidLikelihood<X>, H>:
    ConjugatePrior<X, Fx>
    + UpdatePrior<X, Fx, H>
    + Serialize
    + DeserializeOwned
    + Sync
    + Clone
    + Debug
{
    // Create an empty sufficient statistic for a component
    fn empty_suffstat(&self) -> Fx::Stat;
    // Create a dummy component whose parameters **will be** immediately be
    // overwritten
    //
    // # Note
    // The component must still have the correct dimension for the column. For
    // example, a categorical column must have the correct `k`.
    fn invalid_temp_component(&self) -> Fx;
    // Compute the score of the column for the column reassignment
    fn score_column<I: Iterator<Item = Fx::Stat>>(&self, stats: I) -> f64;
}

use braid_stats::prior::csd::CsdHyper;
use rv::data::CategoricalSuffStat;
use rv::dist::{Categorical, SymmetricDirichlet};

impl BraidPrior<u8, Categorical, CsdHyper> for SymmetricDirichlet {
    fn empty_suffstat(&self) -> CategoricalSuffStat {
        CategoricalSuffStat::new(self.k())
    }

    fn invalid_temp_component(&self) -> Categorical {
        // XXX: This is not a valid distribution. The weights do not sum to 1. I
        // want to leave this invalid, because I want it to show up if we use
        // this someplace we're not supposed to. Anywhere this is supposed to be
        // use used, the bad weights would be immediately overwritten.
        Categorical::new_unchecked(vec![0.0; self.k()])
    }

    fn score_column<I: Iterator<Item = CategoricalSuffStat>>(
        &self,
        stats: I,
    ) -> f64 {
        use special::Gamma;
        let sum_alpha = self.alpha() * self.k() as f64;
        let a = sum_alpha.ln_gamma().0;
        let d = self.alpha().ln_gamma().0 * self.k() as f64;
        stats
            .map(|stat| {
                let b = (sum_alpha + stat.n() as f64).ln_gamma().0;
                let c = stat.counts().iter().fold(0.0, |acc, &ct| {
                    acc + (self.alpha() + ct).ln_gamma().0
                });
                a - b + c - d
            })
            .sum::<f64>()
    }
}

use braid_stats::prior::pg::PgHyper;
use rv::data::PoissonSuffStat;
use rv::dist::{Gamma, Poisson};

#[inline]
fn poisson_zn(shape: f64, rate: f64, stat: &PoissonSuffStat) -> f64 {
    use special::Gamma;
    let shape_n = shape + stat.sum();
    let rate_n = rate + stat.n() as f64;
    let ln_gamma_shape = shape_n.ln_gamma().0;
    let ln_rate = rate_n.ln();
    ln_gamma_shape - shape_n * ln_rate
}

impl BraidPrior<u32, Poisson, PgHyper> for Gamma {
    fn empty_suffstat(&self) -> PoissonSuffStat {
        PoissonSuffStat::new()
    }

    fn invalid_temp_component(&self) -> Poisson {
        Poisson::new_unchecked(1.0)
    }

    fn score_column<I: Iterator<Item = PoissonSuffStat>>(
        &self,
        stats: I,
    ) -> f64 {
        use special::Gamma as _;
        let shape = self.shape();
        let rate = self.rate();
        let z0 = {
            let ln_gamma_shape = shape.ln_gamma().0;
            let ln_rate = rate.ln();
            ln_gamma_shape - shape * ln_rate
        };
        stats
            .map(|stat| {
                let zn = poisson_zn(shape, rate, &stat);
                zn - z0 - stat.sum_ln_fact()
            })
            .sum::<f64>()
    }
}

use braid_stats::prior::nix::NixHyper;
use rv::data::GaussianSuffStat;
use rv::dist::{Gaussian, NormalInvChiSquared};

impl BraidPrior<f64, Gaussian, NixHyper> for NormalInvChiSquared {
    fn empty_suffstat(&self) -> GaussianSuffStat {
        GaussianSuffStat::new()
    }

    fn invalid_temp_component(&self) -> Gaussian {
        Gaussian::standard()
    }

    fn score_column<I: Iterator<Item = GaussianSuffStat>>(
        &self,
        stats: I,
    ) -> f64 {
        use rv::data::DataOrSuffStat;
        let cache = self.ln_m_cache();
        stats
            .map(|stat| {
                let x = DataOrSuffStat::SuffStat(&stat);
                self.ln_m_with_cache(&cache, &x)
            })
            .sum::<f64>()
    }
}

use braid_stats::labeler::Label;
use braid_stats::labeler::Labeler;
use braid_stats::labeler::LabelerPrior;
use braid_stats::labeler::LabelerSuffStat;

impl BraidPrior<Label, Labeler, ()> for LabelerPrior {
    fn empty_suffstat(&self) -> LabelerSuffStat {
        LabelerSuffStat::new()
    }

    fn invalid_temp_component(&self) -> Labeler {
        use braid_stats::SimplexPoint;
        let k = self.pr_world.k();
        // XXX: The simplex point is invalid. But since it *should* be
        // overwritten immediately, it should never cause a problem.
        Labeler::new(0.9, 0.9, SimplexPoint::new_unchecked(vec![0.0; k]))
    }

    fn score_column<I: Iterator<Item = LabelerSuffStat>>(
        &self,
        _stats: I,
    ) -> f64 {
        unimplemented!()
    }
}

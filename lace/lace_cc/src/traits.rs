use std::convert::TryFrom;
use std::fmt::Debug;

use crate::feature::Component;
use lace_consts::rv::experimental::stick_breaking::StickBreaking;
use lace_consts::rv::experimental::stick_breaking::StickBreakingDiscrete;
use lace_consts::rv::experimental::stick_breaking::StickBreakingDiscreteSuffStat;
use lace_consts::rv::experimental::stick_breaking::StickSequence;
use lace_consts::rv::traits::DataOrSuffStat;
use lace_data::Datum;
use lace_data::SparseContainer;
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::prior::sbd::SbdHyper;
use lace_stats::rv::data::{
    BernoulliSuffStat, CategoricalDatum, CategoricalSuffStat, GaussianSuffStat,
    PoissonSuffStat,
};
use lace_stats::rv::dist::{
    Bernoulli, Beta, Categorical, Gamma, Gaussian, NormalInvChiSquared,
    Poisson, SymmetricDirichlet,
};
use lace_stats::rv::traits::{ConjugatePrior, HasSuffStat, Mode, Rv};
use lace_stats::UpdatePrior;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Score accumulation for `finite_cpu` and `slice` row transition kernels.
///
/// Provides two functions to add the scores (log likelihood) of a vector of
/// data to a vector of existing scores.
pub trait AccumScore<X: Clone + Default>: Rv<X> + Sync {
    // XXX: Default implementations can be improved upon by pre-computing
    // normalizers
    fn accum_score(&self, scores: &mut [f64], container: &SparseContainer<X>) {
        use lace_data::AccumScore;
        container.accum_score(scores, &|x| self.ln_f(x))
    }
}

impl<X: CategoricalDatum + Default> AccumScore<X> for Categorical {}
impl AccumScore<u32> for Poisson {}
impl AccumScore<f64> for Gaussian {}
impl AccumScore<bool> for Bernoulli {}
impl AccumScore<usize> for StickBreakingDiscrete {}

/// A Lace-ready datum.
pub trait LaceDatum:
    Sync + Serialize + DeserializeOwned + TryFrom<Datum> + Default + Clone + Debug
{
}

impl<X> LaceDatum for X where
    X: Sync
        + Serialize
        + DeserializeOwned
        + TryFrom<Datum>
        + Default
        + Clone
        + Debug
{
}

/// A Lace-ready datum.
pub trait LaceStat:
    Sync + Serialize + DeserializeOwned + Debug + Clone + PartialEq
{
}
impl<X> LaceStat for X where
    X: Sync + Serialize + DeserializeOwned + Debug + Clone + PartialEq
{
}

/// A Lace-ready likelihood function, f(x).
pub trait LaceLikelihood<X: LaceDatum>:
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
    /// The maximum value the likelihood can take on for this component
    fn ln_f_max(&self) -> Option<f64> {
        self.mode().map(|x| self.ln_f(&x))
    }
}

impl<X, Fx> LaceLikelihood<X> for Fx
where
    X: LaceDatum,
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

/// A Lace-ready prior π(f)
pub trait LacePrior<X: LaceDatum, Fx: LaceLikelihood<X>, H>:
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

impl LacePrior<usize, StickBreakingDiscrete, SbdHyper> for StickBreaking {
    fn empty_suffstat(&self) -> StickBreakingDiscreteSuffStat {
        StickBreakingDiscreteSuffStat::new()
    }

    fn invalid_temp_component(&self) -> StickBreakingDiscrete {
        use lace_stats::rv::dist::UnitPowerLaw;
        // XXX: This is not a valid distribution. The weights do not sum to 1. I
        // want to leave this invalid, because I want it to show up if we use
        // this someplace we're not supposed to. Anywhere this is supposed to be
        // use used, the bad weights would be immediately overwritten.
        StickBreakingDiscrete::new(StickSequence::new(
            UnitPowerLaw::uniform(),
            None,
        ))
    }

    fn score_column<I: Iterator<Item = StickBreakingDiscreteSuffStat>>(
        &self,
        stats: I,
    ) -> f64 {
        let cache = self.ln_m_cache();
        stats
            .map(|stat| {
                self.ln_m_with_cache(&cache, &DataOrSuffStat::SuffStat(&stat))
            })
            .sum::<f64>()
    }
}

impl LacePrior<u8, Categorical, CsdHyper> for SymmetricDirichlet {
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
        let sum_alpha = self.alpha() * self.k() as f64;
        let a = ::special::Gamma::ln_gamma(sum_alpha).0;
        let d = ::special::Gamma::ln_gamma(self.alpha()).0 * self.k() as f64;
        stats
            .map(|stat| {
                let b =
                    ::special::Gamma::ln_gamma(sum_alpha + stat.n() as f64).0;
                let c = stat.counts().iter().fold(0.0, |acc, &ct| {
                    acc + ::special::Gamma::ln_gamma(self.alpha() + ct).0
                });
                a - b + c - d
            })
            .sum::<f64>()
    }
}

#[inline]
fn poisson_zn(shape: f64, rate: f64, stat: &PoissonSuffStat) -> f64 {
    let shape_n = shape + stat.sum();
    let rate_n = rate + stat.n() as f64;
    let ln_gamma_shape = ::special::Gamma::ln_gamma(shape_n).0;
    let ln_rate = rate_n.ln();
    shape_n.mul_add(-ln_rate, ln_gamma_shape)
}

impl LacePrior<u32, Poisson, PgHyper> for Gamma {
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
        let shape = self.shape();
        let rate = self.rate();
        let z0 = {
            let ln_gamma_shape = ::special::Gamma::ln_gamma(shape).0;
            let ln_rate = rate.ln();
            shape.mul_add(-ln_rate, ln_gamma_shape)
        };
        stats
            .map(|stat| {
                let zn = poisson_zn(shape, rate, &stat);
                zn - z0 - stat.sum_ln_fact()
            })
            .sum::<f64>()
    }
}

impl LacePrior<bool, Bernoulli, ()> for Beta {
    fn empty_suffstat(&self) -> BernoulliSuffStat {
        BernoulliSuffStat::new()
    }

    fn invalid_temp_component(&self) -> Bernoulli {
        Bernoulli::uniform()
    }

    fn score_column<I: Iterator<Item = BernoulliSuffStat>>(
        &self,
        stats: I,
    ) -> f64 {
        let cache = <Beta as ConjugatePrior<bool, Bernoulli>>::ln_m_cache(self);
        stats
            .map(|stat| {
                let x = DataOrSuffStat::SuffStat::<bool, Bernoulli>(&stat);
                self.ln_m_with_cache(&cache, &x)
            })
            .sum::<f64>()
    }
}

impl LacePrior<f64, Gaussian, NixHyper> for NormalInvChiSquared {
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
        let cache = self.ln_m_cache();
        stats
            .map(|stat| {
                let x = DataOrSuffStat::SuffStat(&stat);
                self.ln_m_with_cache(&cache, &x)
            })
            .sum::<f64>()
    }
}

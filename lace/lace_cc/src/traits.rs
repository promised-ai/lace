use std::convert::TryFrom;
use std::fmt::Debug;

use crate::feature::Component;
use lace_data::label::Label;
use lace_data::Datum;
use lace_data::SparseContainer;
use lace_stats::labeler::{Labeler, LabelerPrior, LabelerSuffStat};
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
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
impl AccumScore<Label> for Labeler {}
impl AccumScore<u32> for Poisson {}
impl AccumScore<f64> for Gaussian {}
impl AccumScore<bool> for Bernoulli {}

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

#[inline]
fn poisson_zn(shape: f64, rate: f64, stat: &PoissonSuffStat) -> f64 {
    use special::Gamma;
    let shape_n = shape + stat.sum();
    let rate_n = rate + stat.n() as f64;
    let ln_gamma_shape = shape_n.ln_gamma().0;
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
        use special::Gamma as _;
        let shape = self.shape();
        let rate = self.rate();
        let z0 = {
            let ln_gamma_shape = shape.ln_gamma().0;
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
        use lace_stats::rv::data::DataOrSuffStat;
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
        use lace_stats::rv::data::DataOrSuffStat;
        let cache = self.ln_m_cache();
        stats
            .map(|stat| {
                let x = DataOrSuffStat::SuffStat(&stat);
                self.ln_m_with_cache(&cache, &x)
            })
            .sum::<f64>()
    }
}

impl LacePrior<Label, Labeler, ()> for LabelerPrior {
    fn empty_suffstat(&self) -> LabelerSuffStat {
        LabelerSuffStat::new()
    }

    fn invalid_temp_component(&self) -> Labeler {
        use lace_stats::SimplexPoint;
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

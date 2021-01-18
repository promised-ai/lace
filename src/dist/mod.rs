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
    fn empty_suffstat(&self) -> Fx::Stat;
    fn score_column<I: Iterator<Item = Fx::Stat>>(&self, stats: I) -> f64;
}

use braid_stats::prior::csd::CsdHyper;
use rv::data::CategoricalSuffStat;
use rv::dist::{Categorical, SymmetricDirichlet};

impl BraidPrior<u8, Categorical, CsdHyper> for SymmetricDirichlet {
    fn empty_suffstat(&self) -> CategoricalSuffStat {
        CategoricalSuffStat::new(self.k())
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

use braid_stats::prior::ng::NgHyper;
use rv::data::GaussianSuffStat;
use rv::dist::{Gaussian, NormalGamma};

#[inline]
fn normal_gamma_z(r: f64, s: f64, v: f64) -> f64 {
    use special::Gamma;
    let half_v = 0.5 * v;
    (half_v + 0.5).mul_add(std::f64::consts::LN_2, HALF_LN_PI)
        - 0.5f64.mul_add(r.ln(), half_v.mul_add(s.ln(), -half_v.ln_gamma().0))
}

#[inline]
fn normal_gamma_posterior_z(ng: &NormalGamma, stat: &GaussianSuffStat) -> f64 {
    let nf = stat.n() as f64;
    let r = ng.r() + nf;
    let v = ng.v() + nf;
    let m = ng.m().mul_add(ng.r(), stat.sum_x()) / r;
    let s = ng.s()
        + stat.sum_x_sq()
        + ng.r().mul_add(ng.m().powi(2), -r * m.powi(2));
    normal_gamma_z(r, s, v)
}

impl BraidPrior<f64, Gaussian, NgHyper> for NormalGamma {
    fn empty_suffstat(&self) -> GaussianSuffStat {
        GaussianSuffStat::new()
    }

    fn score_column<I: Iterator<Item = GaussianSuffStat>>(
        &self,
        stats: I,
    ) -> f64 {
        let z0 = normal_gamma_z(self.r(), self.s(), self.v());
        stats
            .map(|stat| {
                let zn = normal_gamma_posterior_z(&self, &stat);
                (-(stat.n() as f64)).mul_add(HALF_LN_2PI, zn) - z0
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

    fn score_column<I: Iterator<Item = LabelerSuffStat>>(
        &self,
        _stats: I,
    ) -> f64 {
        unimplemented!()
    }
}

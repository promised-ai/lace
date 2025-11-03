//! Conjugate component data structure
use rv::data::DataOrSuffStat;
use rv::traits::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::cc::feature::Component;
use crate::cc::traits::AccumScore;
use crate::cc::traits::{LaceDatum, LaceLikelihood, LaceStat};
use crate::data::SparseContainer;
use rand::Rng;

/// Maintains a component model and a sufficient statistic capturing the data
/// assigned to the component.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub fx: Fx,
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub stat: Fx::Stat,
    #[serde(skip)]
    pub ln_pp_cache: OnceLock<Pr::PpCache>,
}

impl<X, Fx, Pr> AccumScore<X> for ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn accum_score(&self, scores: &mut [f64], container: &SparseContainer<X>) {
        self.fx.accum_score(scores, container)
    }

    // fn accum_score_par(
    //     &self,
    //     mut scores: &mut [f64],
    //     xs: &[X],
    //     present: &[bool],
    // ) {
    //     self.fx.accum_score_par(&mut scores, &xs, &present)
    // }
}

impl<X, Fx, Pr> ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    /// Create a new ConjugateComponent with no observations
    #[inline]
    pub fn new(fx: Fx) -> Self {
        let stat = fx.empty_suffstat();
        ConjugateComponent {
            fx,
            stat,
            ln_pp_cache: OnceLock::new(),
        }
    }

    /// Return the observations
    #[inline]
    pub fn obs<'a>(&'a self) -> DataOrSuffStat<'a, X, Fx> {
        DataOrSuffStat::SuffStat(&self.stat)
    }

    #[inline]
    pub fn reset_ln_pp_cache(&mut self) {
        self.ln_pp_cache = OnceLock::new()
    }

    #[inline]
    pub fn ln_pp_cache(&self, prior: &Pr) -> &Pr::PpCache {
        self.ln_pp_cache
            .get_or_init(|| prior.ln_pp_cache(&self.obs()))
    }
}

impl<X, Fx, Pr> HasDensity<X> for ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn ln_f(&self, x: &X) -> f64 {
        self.fx.ln_f(x)
    }
}

impl<X, Fx, Pr> Sampleable<X> for ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        self.fx.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        self.fx.sample(n, &mut rng)
    }
}

impl<X, Fx, Pr> Mode<X> for ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X> + Mode<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn mode(&self) -> Option<X> {
        self.fx.mode()
    }
}

impl<X, Fx, Pr> Entropy for ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X> + Entropy,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn entropy(&self) -> f64 {
        self.fx.entropy()
    }
}

impl<X, Fx, Pr> SuffStat<X> for ConjugateComponent<X, Fx, Pr>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn n(&self) -> usize {
        self.stat.n()
    }

    fn observe(&mut self, x: &X) {
        self.reset_ln_pp_cache();
        self.stat.observe(x);
    }

    fn forget(&mut self, x: &X) {
        self.reset_ln_pp_cache();
        self.stat.forget(x);
    }

    fn merge(&mut self, other: Self) {
        self.stat.merge(other.stat);
    }
}

impl<X, Fx, Pr> From<ConjugateComponent<X, Fx, Pr>> for Component
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn from(cpnt: ConjugateComponent<X, Fx, Pr>) -> Component {
        cpnt.fx.into()
    }
}

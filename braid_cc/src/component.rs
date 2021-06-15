//! Conjugate component data structure
use braid_data::SparseContainer;
use once_cell::sync::OnceCell;
use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::feature::Component;
use crate::traits::AccumScore;
use crate::traits::{BraidDatum, BraidLikelihood, BraidStat};

/// Maintains a component model and a sufficient statistic capturing the data
/// assigned to the component.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub fx: Fx,
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub stat: Fx::Stat,
    #[serde(skip)]
    pub ln_pp_cache: OnceCell<Pr::LnPpCache>,
}

impl<X, Fx, Pr> AccumScore<X> for ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn accum_score(
        &self,
        mut scores: &mut [f64],
        container: &SparseContainer<X>,
    ) {
        self.fx.accum_score(&mut scores, container)
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
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    /// Create a new ConjugateComponent with no observations
    #[inline]
    pub fn new(fx: Fx) -> Self {
        let stat = fx.empty_suffstat();
        ConjugateComponent {
            fx,
            stat,
            ln_pp_cache: OnceCell::new(),
        }
    }

    /// Return the observations
    #[inline]
    pub fn obs(&self) -> DataOrSuffStat<X, Fx> {
        DataOrSuffStat::SuffStat(&self.stat)
    }

    #[inline]
    pub fn reset_ln_pp_cache(&mut self) {
        self.ln_pp_cache = OnceCell::new()
    }

    #[inline]
    pub fn ln_pp_cache(&self, prior: &Pr) -> &Pr::LnPpCache {
        self.ln_pp_cache
            .get_or_init(|| prior.ln_pp_cache(&self.obs()))
    }
}

impl<X, Fx, Pr> Rv<X> for ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn ln_f(&self, x: &X) -> f64 {
        self.fx.ln_f(x)
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        self.fx.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        self.fx.sample(n, &mut rng)
    }
}

impl<X, Fx, Pr> Mode<X> for ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X> + Mode<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn mode(&self) -> Option<X> {
        self.fx.mode()
    }
}

impl<X, Fx, Pr> Entropy for ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X> + Entropy,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn entropy(&self) -> f64 {
        self.fx.entropy()
    }
}

impl<X, Fx, Pr> SuffStat<X> for ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
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
}

impl<X, Fx, Pr> From<ConjugateComponent<X, Fx, Pr>> for Component
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn from(cpnt: ConjugateComponent<X, Fx, Pr>) -> Component {
        cpnt.fx.into()
    }
}

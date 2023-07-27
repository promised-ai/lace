use crate::rv::data::DataOrSuffStat;
use crate::rv::dist::{
    Categorical, CategoricalError, Gamma, SymmetricDirichlet,
};
use crate::rv::traits::{
    ConjugatePrior, Entropy, HasSuffStat, Mode, Rv, SuffStat,
};
use crate::UpdatePrior;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord,
)]
pub struct DpDiscreteSuffStat {
    n: usize,
    counts: Vec<usize>,
}

impl DpDiscreteSuffStat {
    pub fn new() -> Self {
        Self {
            n: 0,
            counts: Vec::new(),
        }
    }
}

impl SuffStat<usize> for DpDiscreteSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn forget(&mut self, x: &usize) {
        if self.n == 1 {
            self.n = 0;
            self.counts = Vec::new();
        } else if self.counts[*x] == 1 {
            // remove this
            self.counts.remove(*x);
        } else {
            // could should be greater than 1
            self.counts[*x] -= 1;
        }
    }

    fn observe(&mut self, x: &usize) {
        unimplemented!()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DpDiscrete {
    k: usize,
    m: usize,
    categorical: Categorical,
}

impl DpDiscrete {
    pub fn new(k: usize, weights: &[f64]) -> Result<Self, CategoricalError> {
        let categorical = Categorical::new(weights)?;
        Ok(Self {
            k,
            m: categorical.k() - k,
            categorical,
        })
    }

    pub fn uniform(k: usize, m: usize) -> Self {
        Self {
            k,
            m,
            categorical: Categorical::uniform(k + m),
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn ln_weights(&self) -> &[f64] {
        self.categorical.ln_weights()
    }

    pub fn len(&self) -> usize {
        self.k + self.m
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Rv<usize> for DpDiscrete {
    fn ln_f(&self, x: &usize) -> f64 {
        let ln_weights = self.categorical.ln_weights();
        if *x >= self.k {
            ln_weights[self.k]
        } else {
            ln_weights[*x]
        }
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.categorical.draw(rng)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct StickBreaking {
    // k: usize,
    alpha: f64,
}

impl StickBreaking {
    pub fn new_unchecked(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Default for StickBreaking {
    fn default() -> Self {
        StickBreaking::new_unchecked(1.0)
    }
}

impl HasSuffStat<usize> for DpDiscrete {
    type Stat = DpDiscreteSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        DpDiscreteSuffStat::new()
    }
}

impl Rv<DpDiscrete> for StickBreaking {
    fn ln_f(&self, x: &DpDiscrete) -> f64 {
        unimplemented!()
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> DpDiscrete {
        unimplemented!()
    }
}

impl Rv<StickBreaking> for Gamma {
    fn ln_f(&self, x: &StickBreaking) -> f64 {
        self.ln_f(&x.alpha)
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> StickBreaking {
        let alpha: f64 = self.draw(rng);
        StickBreaking { alpha }
    }
}

impl ConjugatePrior<usize, DpDiscrete> for StickBreaking {
    type LnMCache = ();
    type LnPpCache = ();
    type Posterior = StickBreaking;

    fn ln_m_cache(&self) -> Self::LnMCache {
        unimplemented!()
    }

    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<usize, DpDiscrete>,
    ) -> Self::LnPpCache {
        unimplemented!()
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
        unimplemented!()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::LnMCache,
        x: &DataOrSuffStat<usize, DpDiscrete>,
    ) -> f64 {
        unimplemented!()
    }

    fn posterior(
        &self,
        x: &DataOrSuffStat<usize, DpDiscrete>,
    ) -> Self::Posterior {
        unimplemented!()
    }
}

impl Entropy for DpDiscrete {
    fn entropy(&self) -> f64 {
        self.categorical.entropy()
    }
}

impl Mode<usize> for DpDiscrete {
    fn mode(&self) -> Option<usize> {
        self.categorical.mode()
    }
}

impl UpdatePrior<usize, DpDiscrete, Gamma> for StickBreaking {
    fn update_prior<R: rand::Rng>(
        &mut self,
        components: &[&DpDiscrete],
        hyper: &Gamma,
        rng: &mut R,
    ) -> f64 {
        let k = components[0].k;

        let ln_prior = |alpha: &f64| {
            let dir = SymmetricDirichlet::new_unchecked(*alpha, k);
            components
                .iter()
                .map(|cpnt| dir.ln_f(&cpnt.categorical))
                .sum::<f64>()
        };

        let res = crate::mh::mh_prior(
            self.alpha,
            ln_prior,
            |r| hyper.draw(r),
            200,
            rng,
        );

        self.alpha = res.x;

        res.score_x
    }
}

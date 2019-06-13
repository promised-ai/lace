use std::convert::Into;

use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::cc::Component;
use crate::dist::traits::AccumScore;
use crate::dist::{BraidDatum, BraidLikelihood, BraidStat};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjugateComponent<X, Fx>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
{
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub fx: Fx,
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub stat: Fx::Stat,
}

impl<X, Fx> AccumScore<X> for ConjugateComponent<X, Fx>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
{
    fn accum_score(&self, mut scores: &mut [f64], xs: &[X], present: &[bool]) {
        self.fx.accum_score(&mut scores, &xs, &present)
    }

    fn accum_score_par(
        &self,
        mut scores: &mut [f64],
        xs: &[X],
        present: &[bool],
    ) {
        self.fx.accum_score_par(&mut scores, &xs, &present)
    }
}

impl<X, Fx> ConjugateComponent<X, Fx>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
{
    /// Create a new ConjugateComponent with no observations
    pub fn new(fx: Fx) -> Self {
        let stat = fx.empty_suffstat();
        ConjugateComponent { fx, stat }
    }

    /// Return the observations
    pub fn obs(&self) -> DataOrSuffStat<X, Fx> {
        DataOrSuffStat::SuffStat(&self.stat)
    }
}

impl<X, Fx> Rv<X> for ConjugateComponent<X, Fx>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
{
    fn ln_f(&self, x: &X) -> f64 {
        self.fx.ln_f(&x)
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        self.fx.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        self.fx.sample(n, &mut rng)
    }
}

impl<X, Fx> SuffStat<X> for ConjugateComponent<X, Fx>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
{
    fn n(&self) -> usize {
        self.stat.n()
    }

    fn observe(&mut self, x: &X) {
        self.stat.observe(&x);
    }

    fn forget(&mut self, x: &X) {
        self.stat.forget(&x);
    }
}

impl<X, Fx> Into<Component> for ConjugateComponent<X, Fx>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
{
    fn into(self) -> Component {
        self.fx.into()
    }
}

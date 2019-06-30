use rand::Rng;
use rv::data::{CategoricalDatum, DataOrSuffStat};
use rv::dist::{Categorical, Dirichlet, InvGamma, SymmetricDirichlet};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::defaults;
use crate::mh::mh_prior;
use crate::UpdatePrior;

/// Symmetric Dirichlet prior for the categorical α parameter
///
/// If x ~ Categorical(**w**), where **w**=[w<sub>1</sub>, ..., w<sub>k</sub>]),
/// then **w** ~ Dirichlet([α, ..., α]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct Csd {
    /// Symmetric Dirichlet prior on weights
    pub symdir: SymmetricDirichlet,
    /// Hyper-prior on Symmetric Dirichlet α
    pub hyper: CsdHyper,
}

impl Csd {
    pub fn new(alpha: f64, k: usize, hyper: CsdHyper) -> Self {
        Csd {
            symdir: SymmetricDirichlet::new(alpha, k).unwrap(),
            hyper,
        }
    }

    /// Default `Csd` for Geweke testing
    pub fn geweke(k: usize) -> Self {
        Csd {
            symdir: SymmetricDirichlet::new(1.0, k).unwrap(),
            hyper: CsdHyper::geweke(),
        }
    }

    /// Draw the prior from the hyper-prior
    pub fn from_hyper(
        k: usize,
        hyper: CsdHyper,
        mut rng: &mut impl Rng,
    ) -> Self {
        hyper.draw(k, &mut rng)
    }

    /// Build a vague hyper-prior given `k` and draws the prior from that
    pub fn vague(k: usize, mut rng: &mut impl Rng) -> Self {
        let hyper = CsdHyper::new(k as f64 + 1.0, 1.0);
        hyper.draw(k, &mut rng)
    }
}

impl Rv<Categorical> for Csd {
    fn ln_f(&self, model: &Categorical) -> f64 {
        self.symdir.ln_f(model)
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        self.symdir.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<Categorical> {
        self.symdir.sample(n, &mut rng)
    }
}

impl<X: CategoricalDatum> ConjugatePrior<X, Categorical> for Csd {
    type Posterior = Dirichlet;
    fn posterior(&self, x: &DataOrSuffStat<X, Categorical>) -> Dirichlet {
        self.symdir.posterior(&x)
    }

    fn ln_m(&self, x: &DataOrSuffStat<X, Categorical>) -> f64 {
        self.symdir.ln_m(&x)
    }

    fn ln_pp(&self, y: &X, x: &DataOrSuffStat<X, Categorical>) -> f64 {
        self.symdir.ln_pp(y, x)
    }
}

impl<X: CategoricalDatum> UpdatePrior<X, Categorical> for Csd {
    fn update_prior<R: Rng>(
        &mut self,
        components: &Vec<&Categorical>,
        mut rng: &mut R,
    ) {
        let new_alpha = {
            let draw = |mut rng: &mut R| self.hyper.pr_alpha.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |alpha: &f64| {
                let h = self.hyper.clone();
                let csd = Csd::new(*alpha, self.symdir.k, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + csd.ln_f(&cpnt))
            };
            mh_prior(
                self.symdir.alpha,
                f,
                draw,
                defaults::MH_PRIOR_ITERS,
                &mut rng,
            )
        };
        self.symdir.alpha = new_alpha;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct CsdHyper {
    pub pr_alpha: InvGamma,
}

impl Default for CsdHyper {
    fn default() -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(1.0, 1.0).unwrap(),
        }
    }
}

impl CsdHyper {
    pub fn new(shape: f64, rate: f64) -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(shape, rate).unwrap(),
        }
    }

    /// A restrictive prior to confine Geweke.
    ///
    /// Since the geweke test seeks to draw samples from the joint of the prior
    /// and the data, p(x, θ), and since θ is indluenced by the hyper-prior, if
    /// the hyper parameters are not tight, the data can go crazy and cause a
    /// bunch of math errors.
    pub fn geweke() -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(30.0, 29.0).unwrap(),
        }
    }

    /// α ~ Gamma(k + 1, 1)
    pub fn vague(k: usize) -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(k as f64 + 1.0, 1.0).unwrap(),
        }
    }

    /// Draw a `Csd` from the hyper-prior
    pub fn draw(&self, k: usize, mut rng: &mut impl Rng) -> Csd {
        // SymmetricDirichlet::new(self.pr_alpha.draw(&mut rng), k);
        let alpha = self.pr_alpha.draw(&mut rng);
        Csd::new(alpha, k, self.clone())
    }
}

use rand::Rng;
use rv::data::CategoricalDatum;
use rv::dist::{Categorical, InvGamma, SymmetricDirichlet};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::mh::mh_prior;
use crate::UpdatePrior;

/// Symmetric Dirichlet prior for the categorical α parameter
///
/// If x ~ Categorical(**w**), where **w**=[w<sub>1</sub>, ..., w<sub>k</sub>]),
/// then **w** ~ Dirichlet([α, ..., α]).
// #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
// pub struct Csd {
//     /// Symmetric Dirichlet prior on weights
//     pub symdir: SymmetricDirichlet,
// }

/// Default `Csd` for Geweke testing
pub fn geweke(k: usize) -> SymmetricDirichlet {
    SymmetricDirichlet::new_unchecked(1.0, k)
}

/// Draw the prior from the hyper-prior
pub fn from_hyper(
    k: usize,
    hyper: CsdHyper,
    mut rng: &mut impl Rng,
) -> SymmetricDirichlet {
    hyper.draw(k, &mut rng)
}

/// Build a vague hyper-prior given `k` and draws the prior from that
pub fn vague(k: usize, mut rng: &mut impl Rng) -> SymmetricDirichlet {
    let hyper = CsdHyper::new(k as f64 + 1.0, 1.0);
    hyper.draw(k, &mut rng)
}

// impl Csd {
//     pub fn new(alpha: f64, k: usize, hyper: CsdHyper) -> Self {
//         Csd {
//             symdir: SymmetricDirichlet::new(alpha, k).unwrap(),
//         }
//     }

//     /// Default `Csd` for Geweke testing
//     pub fn geweke(k: usize) -> Self {
//         Csd {
//             symdir: SymmetricDirichlet::new(1.0, k).unwrap(),
//         }
//     }

//     /// Draw the prior from the hyper-prior
//     pub fn from_hyper(
//         k: usize,
//         hyper: CsdHyper,
//         mut rng: &mut impl Rng,
//     ) -> Self {
//         hyper.draw(k, &mut rng)
//     }

//     /// Build a vague hyper-prior given `k` and draws the prior from that
//     pub fn vague(k: usize, mut rng: &mut impl Rng) -> Self {
//         let hyper = CsdHyper::new(k as f64 + 1.0, 1.0);
//         hyper.draw(k, &mut rng)
//     }
// }

// impl Rv<Categorical> for Csd {
//     fn ln_f(&self, model: &Categorical) -> f64 {
//         self.symdir.ln_f(model)
//     }

//     fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
//         self.symdir.draw(&mut rng)
//     }

//     fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<Categorical> {
//         self.symdir.sample(n, &mut rng)
//     }
// }

// impl<X: CategoricalDatum> ConjugatePrior<X, Categorical> for Csd {
//     type Posterior = Dirichlet;
//     type LnMCache =
//         <SymmetricDirichlet as ConjugatePrior<X, Categorical>>::LnMCache;
//     type LnPpCache =
//         <SymmetricDirichlet as ConjugatePrior<X, Categorical>>::LnPpCache;

//     #[inline]
//     fn posterior(&self, x: &DataOrSuffStat<X, Categorical>) -> Dirichlet {
//         self.symdir.posterior(&x)
//     }

//     #[inline]
//     fn ln_m_cache(&self) -> Self::LnMCache {
//         <SymmetricDirichlet as ConjugatePrior<X, Categorical>>::ln_m_cache(
//             &self.symdir,
//         )
//     }

//     #[inline]
//     fn ln_m_with_cache(
//         &self,
//         cache: &Self::LnMCache,
//         x: &DataOrSuffStat<X, Categorical>,
//     ) -> f64 {
//         self.symdir.ln_m_with_cache(cache, x)
//     }

//     #[inline]
//     fn ln_pp_cache(
//         &self,
//         x: &DataOrSuffStat<X, Categorical>,
//     ) -> Self::LnPpCache {
//         self.symdir.ln_pp_cache(x)
//     }

//     #[inline]
//     fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &X) -> f64 {
//         self.symdir.ln_pp_with_cache(cache, y)
//     }
// }

impl<X: CategoricalDatum> UpdatePrior<X, Categorical, CsdHyper>
    for SymmetricDirichlet
{
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Categorical],
        hyper: &CsdHyper,
        mut rng: &mut R,
    ) -> f64 {
        use special::Gamma;
        let mh_result = {
            let draw = |mut rng: &mut R| hyper.pr_alpha.draw(&mut rng);

            let k = self.k();
            let kf = k as f64;

            let f = |alpha: &f64| {
                // Pre-compute costly gamma_ln functions
                let sum_ln_gamma = (*alpha).ln_gamma().0 * kf;
                let ln_gamma_sum = (*alpha * kf).ln_gamma().0;
                let am1 = alpha - 1.0;

                components
                    .iter()
                    .map(|cpnt| {
                        let term = cpnt
                            .ln_weights()
                            .iter()
                            .map(|&ln_w| am1 * ln_w)
                            .sum::<f64>();
                        term - (sum_ln_gamma - ln_gamma_sum)
                    })
                    .sum::<f64>()
            };

            mh_prior(
                self.alpha(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            )
        };

        self.set_alpha(mh_result.x).unwrap();
        mh_result.score_x + hyper.pr_alpha.ln_f(&mh_result.x)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    pub fn draw(&self, k: usize, mut rng: &mut impl Rng) -> SymmetricDirichlet {
        // SymmetricDirichlet::new(self.pr_alpha.draw(&mut rng), k);
        let alpha = self.pr_alpha.draw(&mut rng);
        SymmetricDirichlet::new_unchecked(alpha, k)
    }
}

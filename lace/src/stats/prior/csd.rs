use rand::Rng;
use rv::data::CategoricalDatum;
use rv::dist::Categorical;
use rv::dist::InvGamma;
use rv::dist::SymmetricDirichlet;
use rv::traits::*;
use serde::Deserialize;
use serde::Serialize;

use crate::stats::mh::mh_prior;
use crate::stats::UpdatePrior;

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
pub fn vague(k: usize) -> SymmetricDirichlet {
    SymmetricDirichlet::new_unchecked(0.5, k)
}

impl<X: CategoricalDatum> UpdatePrior<X, Categorical, CsdHyper>
    for SymmetricDirichlet
{
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Categorical],
        hyper: &CsdHyper,
        rng: &mut R,
    ) -> f64 {
        let mh_result = {
            let k = self.k();
            let kf = k as f64;

            let loglike = |alpha: &f64| {
                // Pre-compute costly gamma_ln functions
                let sum_ln_gamma = special::Gamma::ln_gamma(*alpha).0 * kf;
                let ln_gamma_sum = special::Gamma::ln_gamma(alpha * kf).0;
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
                loglike,
                |rng| hyper.pr_alpha.draw(rng),
                crate::consts::MH_PRIOR_ITERS,
                rng,
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

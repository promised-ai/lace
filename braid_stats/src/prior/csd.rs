use rand::Rng;
use rv::data::CategoricalDatum;
use rv::dist::{Categorical, InvGamma, SymmetricDirichlet};
use rv::traits::*;
use serde::{Deserialize, Serialize};

// use crate::mh::mh_prior;
// use crate::mh::mh_slice;
use crate::mh::mh_symrw_adaptive;
use crate::UpdatePrior;

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
            // let draw = |mut rng: &mut R| hyper.pr_alpha.draw(&mut rng);

            let k = self.k();
            let kf = k as f64;

            // let f = |alpha: &f64| {
            //     // Pre-compute costly gamma_ln functions
            //     let sum_ln_gamma = (*alpha).ln_gamma().0 * kf;
            //     let ln_gamma_sum = (*alpha * kf).ln_gamma().0;
            //     let am1 = alpha - 1.0;

            //     components
            //         .iter()
            //         .map(|cpnt| {
            //             let term = cpnt
            //                 .ln_weights()
            //                 .iter()
            //                 .map(|&ln_w| am1 * ln_w)
            //                 .sum::<f64>();
            //             term - (sum_ln_gamma - ln_gamma_sum)
            //         })
            //         .sum::<f64>()
            // };

            // mh_prior(
            //     self.alpha(),
            //     f,
            //     draw,
            //     braid_consts::MH_PRIOR_ITERS,
            //     &mut rng,
            // )

            let f = |alpha: f64| {
                // Pre-compute costly gamma_ln functions
                let sum_ln_gamma = alpha.ln_gamma().0 * kf;
                let ln_gamma_sum = (alpha * kf).ln_gamma().0;
                let am1 = alpha - 1.0;

                let sum = components
                    .iter()
                    .map(|cpnt| {
                        let term = cpnt
                            .ln_weights()
                            .iter()
                            .map(|&ln_w| am1 * ln_w)
                            .sum::<f64>();
                        term - (sum_ln_gamma - ln_gamma_sum)
                    })
                    .sum::<f64>();
                sum + hyper.pr_alpha.ln_f(&alpha)
            };

            // println!("= CSD alpha");
            mh_symrw_adaptive(
                self.alpha(),
                hyper.pr_alpha.mean().unwrap_or(1_f64).sqrt(),
                hyper.pr_alpha.variance().unwrap_or(10_f64).sqrt() / 10.0,
                braid_consts::MH_PRIOR_ITERS,
                f,
                (0.0, std::f64::INFINITY),
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

use braid_utils::{mean, var};
use rand::Rng;
// use rv::data::DataOrSuffStat;
use rv::dist::{Gamma, Gaussian, NormalGamma};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::mh::mh_prior;
use crate::UpdatePrior;

/// Default prior parameters for Geweke testing
pub fn geweke() -> NormalGamma {
    NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0)
}

/// Creates an `Ng` with a vague hyper-prior derived from the data
pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> NormalGamma {
    NgHyper::from_data(&xs).draw(&mut rng)
}

/// Draws an `Ng` given a hyper-prior
pub fn from_hyper(hyper: NgHyper, mut rng: &mut impl Rng) -> NormalGamma {
    hyper.draw(&mut rng)
}

impl UpdatePrior<f64, Gaussian, NgHyper> for NormalGamma {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Gaussian],
        hyper: &NgHyper,
        mut rng: &mut R,
    ) -> f64 {
        let new_m: f64;
        let new_r: f64;
        let new_s: f64;
        let new_v: f64;
        let mut ln_prior = 0.0;

        // TODO: We could get more aggressive with the catching. We could pre-compute the
        // ln(sigma) for each component.
        struct Gauss {
            mu: f64,
            sigma: f64,
            ln_sigma: f64,
        }

        let gausses: Vec<Gauss> = components
            .iter()
            .map(|cpnt| Gauss {
                mu: cpnt.mu(),
                sigma: cpnt.sigma(),
                ln_sigma: cpnt.sigma().ln(),
            })
            .collect();

        // TODO: Can we macro these away?
        {
            let draw = |mut rng: &mut R| hyper.pr_m.draw(&mut rng);
            let rs = self.r().recip().sqrt();
            let ln_rs = rs.ln();

            let f = |m: &f64| {
                let errs = gausses
                    .iter()
                    .map(|cpnt| {
                        let sigma = cpnt.sigma * rs;
                        cpnt.ln_sigma
                            + ln_rs
                            + 0.5 * ((m - cpnt.mu) / sigma).powi(2)
                    })
                    .sum::<f64>();
                -errs
            };
            let result = mh_prior(
                self.m(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            new_m = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_m.ln_f(&new_m);
        }

        self.set_m(new_m).unwrap();

        // update r
        {
            let draw = |mut rng: &mut R| hyper.pr_r.draw(&mut rng);
            let f = |r: &f64| {
                let rs = (*r).recip().sqrt();
                let ln_rs = rs.ln();
                let m = self.m();
                let errs = gausses
                    .iter()
                    .map(|cpnt| {
                        let sigma = cpnt.sigma * rs;
                        cpnt.ln_sigma
                            + ln_rs
                            + 0.5 * ((m - cpnt.mu) / sigma).powi(2)
                    })
                    .sum::<f64>();
                -errs
            };

            let result = mh_prior(
                self.r(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            new_r = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_r.ln_f(&new_r);
        }

        self.set_r(new_r).unwrap();

        // update s
        {
            let shape = 0.5 * self.v();
            let draw = |mut rng: &mut R| hyper.pr_s.draw(&mut rng);
            let f = |s: &f64| {
                // we can save a good chunk of time by never computing the
                // gamma(shape) term because we don't need it because we're not
                // re-sampling shape
                let rate = 0.5 * s;
                let ln_rate = rate.ln();

                gausses
                    .iter()
                    .map(|cpnt| {
                        let rho = cpnt.sigma.recip().powi(2);
                        let ln_rho = -2.0 * cpnt.ln_sigma;
                        shape * ln_rate + (shape - 1.0) * ln_rho - (rate * rho)
                    })
                    .sum::<f64>()
            };
            let result = mh_prior(
                self.s(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            new_s = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_s.ln_f(&new_s);
        }

        self.set_s(new_s).unwrap();

        // update v
        {
            use special::Gamma;
            let draw = |mut rng: &mut R| hyper.pr_v.draw(&mut rng);

            let rate = 0.5 * self.s();
            let ln_rate = rate.ln();
            let f = |v: &f64| {
                let shape = 0.5 * v;
                let ln_gamma_shape = shape.ln_gamma().0;
                gausses
                    .iter()
                    .map(|cpnt| {
                        let rho = cpnt.sigma.recip().powi(2);
                        let ln_rho = -2.0 * cpnt.ln_sigma;
                        shape * ln_rate - ln_gamma_shape
                            + (shape - 1.0) * ln_rho
                            - (rate * rho)
                    })
                    .sum::<f64>()
            };

            let result = mh_prior(
                self.v(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            new_v = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_v.ln_f(&new_v);
        }

        self.set_v(new_v).unwrap();

        ln_prior
    }
}

/// Hyper-prior for Normal Gamma (`Ng`)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NgHyper {
    /// Prior on `m`
    pub pr_m: Gaussian,
    /// Prior on `r`
    pub pr_r: Gamma,
    /// Prior on `s`
    pub pr_s: Gamma,
    /// Prior on `v`
    pub pr_v: Gamma,
}

impl Default for NgHyper {
    fn default() -> Self {
        NgHyper {
            pr_m: Gaussian::new(0.0, 1.0).unwrap(),
            pr_r: Gamma::new(2.0, 2.0).unwrap(),
            pr_s: Gamma::new(2.0, 2.0).unwrap(),
            pr_v: Gamma::new(2.0, 2.0).unwrap(),
        }
    }
}

impl NgHyper {
    pub fn new(pr_m: Gaussian, pr_r: Gamma, pr_s: Gamma, pr_v: Gamma) -> Self {
        NgHyper {
            pr_m,
            pr_r,
            pr_s,
            pr_v,
        }
    }

    /// A restrictive prior to confine Geweke.
    ///
    /// Since the geweke test seeks to draw samples from the joint of the prior
    /// and the data, p(x, θ), and since θ is indluenced by the hyper-prior, if
    /// the hyper parameters are not tight, the data can go crazy and cause a
    /// bunch of math errors.
    pub fn geweke() -> Self {
        NgHyper {
            pr_m: Gaussian::new(0.0, 0.1).unwrap(),
            pr_r: Gamma::new(40.0, 4.0).unwrap(),
            pr_s: Gamma::new(40.0, 4.0).unwrap(),
            pr_v: Gamma::new(40.0, 4.0).unwrap(),
        }
    }

    /// Vague prior from the data.
    pub fn from_data(xs: &[f64]) -> Self {
        // How the prior is set up:
        // - The expected mean should be the mean of the data
        // - The stddev of the mean should be stddev of the data
        // - The expected sttdev should be stddev(x) / ln(n)
        // - The sttdev of stddev should be stddev(x) / ln(n)
        // let ln_n = (xs.len() as f64).ln();
        let ln_n = (xs.len() as f64).sqrt();
        let m = mean(xs);
        let v = var(xs);
        let s = v.sqrt();
        NgHyper {
            pr_m: Gaussian::new(m, s).unwrap(),
            pr_r: Gamma::new(ln_n, ln_n.sqrt()).unwrap(),
            pr_s: Gamma::new(2.0 * s, 2.0).unwrap(),
            pr_v: Gamma::new(ln_n, 2.0 + ln_n).unwrap(),
        }
    }

    /// Draw an `Ng` from the hyper
    pub fn draw(&self, mut rng: &mut impl Rng) -> NormalGamma {
        NormalGamma::new_unchecked(
            self.pr_m.draw(&mut rng),
            self.pr_r.draw(&mut rng),
            self.pr_s.draw(&mut rng),
            self.pr_v.draw(&mut rng),
        )
    }
}

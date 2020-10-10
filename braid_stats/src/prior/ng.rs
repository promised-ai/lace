use braid_utils::{mean, var};
use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::dist::{Gamma, Gaussian, NormalGamma};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::mh::mh_prior;
use crate::UpdatePrior;

const HALF_LN_2_PI: f64 =
    0.918938533204672741780329736405617639861397473637783412817151540;

/// Normmal, Inverse-Gamma prior for Normal/Gassuain data
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Ng {
    /// Prior on parameters in N(μ, σ)
    pub ng: NormalGamma,
    /// Hyper-prior on `NormalGamma` Parameters
    pub hyper: NigHyper,
}

impl Ng {
    pub fn new(m: f64, r: f64, s: f64, v: f64, hyper: NigHyper) -> Self {
        Ng {
            ng: NormalGamma::new(m, r, s, v).expect("invalid Ng::new params"),
            hyper,
        }
    }

    /// Default prior parameters for Geweke testing
    pub fn geweke() -> Self {
        Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::geweke())
    }

    // TODO: implement for f32 and f64 data
    /// Creates an `Ng` with a vague hyper-prior derived from the data
    pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> Self {
        NigHyper::from_data(&xs).draw(&mut rng)
    }

    /// Draws an `Ng` given a hyper-prior
    pub fn from_hyper(hyper: NigHyper, mut rng: &mut impl Rng) -> Self {
        hyper.draw(&mut rng)
    }
}

impl Rv<Gaussian> for Ng {
    fn ln_f(&self, model: &Gaussian) -> f64 {
        self.ng.ln_f(&model)
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        self.ng.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<Gaussian> {
        self.ng.sample(n, &mut rng)
    }
}

impl ConjugatePrior<f64, Gaussian> for Ng {
    type Posterior = NormalGamma;
    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> NormalGamma {
        use rv::data::GaussianSuffStat;
        use std::panic::catch_unwind;
        match catch_unwind(|| self.ng.posterior(&x)) {
            Ok(ng) => ng,
            Err(_) => {
                let (suffstat, origin) = match x {
                    DataOrSuffStat::SuffStat(stat) => {
                        ((*stat).to_owned(), "stat")
                    }
                    DataOrSuffStat::Data(data) => {
                        let mut stat = GaussianSuffStat::new();
                        stat.observe_many(&data);
                        (stat, "data")
                    }
                    DataOrSuffStat::None => (GaussianSuffStat::new(), "none"),
                };
                panic!(
                    "Failed to generate posterior from self `{:?}`. \
                     \nInput sufficient statistics: {:?}",
                    self, suffstat
                );
            }
        }
    }

    fn ln_m(&self, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        self.ng.ln_m(&x)
    }

    fn ln_pp(&self, y: &f64, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        self.ng.ln_pp(&y, &x)
    }
}

impl UpdatePrior<f64, Gaussian> for Ng {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Gaussian],
        mut rng: &mut R,
    ) -> f64 {
        let new_m: f64;
        let new_r: f64;
        let new_s: f64;
        let new_v: f64;
        let mut ln_prior = 0.0;

        // TODO: moving to private fields in rv has increased the runtime of the
        // Gaussian benchmark 5%.
        // TODO: Can we macro these away?
        {
            let draw = |mut rng: &mut R| self.hyper.pr_m.draw(&mut rng);
            let f = |m: &f64| {
                let rs = self.ng.r().recip().sqrt();
                let errs = components
                    .iter()
                    .map(|cpnt| {
                        let sigma = cpnt.sigma() * rs;
                        sigma.ln() + 0.5 * ((m - cpnt.mu()) / sigma).powi(2)
                    })
                    .sum::<f64>();
                -errs
            };
            let result = mh_prior(
                self.ng.m(),
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
            ln_prior += result.score_x + self.hyper.pr_m.ln_f(&new_m);
        }

        self.ng =
            NormalGamma::new(new_m, self.ng.r(), self.ng.s(), self.ng.v())
                .unwrap();

        // update r
        {
            let draw = |mut rng: &mut R| self.hyper.pr_r.draw(&mut rng);
            let f = |r: &f64| {
                let rs = (*r).recip().sqrt();
                let m = self.ng.m();
                let errs = components
                    .iter()
                    .map(|cpnt| {
                        let sigma = cpnt.sigma() * rs;
                        sigma.ln() + 0.5 * ((m - cpnt.mu()) / sigma).powi(2)
                    })
                    .sum::<f64>();
                -errs
            };

            let result = mh_prior(
                self.ng.r(),
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
            ln_prior += result.score_x + self.hyper.pr_r.ln_f(&new_r);
        }

        self.ng =
            NormalGamma::new(self.ng.m(), new_r, self.ng.s(), self.ng.v())
                .unwrap();

        // update s
        {
            let draw = |mut rng: &mut R| self.hyper.pr_s.draw(&mut rng);
            let f = |s: &f64| {
                let alpha = 0.5 * self.ng.v();
                let beta = 0.5 * s;
                components
                    .iter()
                    .map(|cpnt| {
                        let rho = cpnt.sigma().recip().powi(2);
                        Gamma::new_unchecked(alpha, beta).ln_f(&rho)
                    })
                    .sum::<f64>()
            };
            let result = mh_prior(
                self.ng.s(),
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
            ln_prior += result.score_x + self.hyper.pr_s.ln_f(&new_s);
        }

        self.ng =
            NormalGamma::new(self.ng.m(), self.ng.r(), new_s, self.ng.v())
                .unwrap();

        // update v
        {
            let draw = |mut rng: &mut R| self.hyper.pr_v.draw(&mut rng);

            let f = |v: &f64| {
                // let h = self.hyper.clone();
                // let ng = Ng::new(self.ng.m(), self.ng.r(), self.ng.s(), *v, h);
                // components
                //     .iter()
                //     .fold(0.0, |logf, cpnt| logf + ng.ln_f(&cpnt))
                let alpha = 0.5 * v;
                let beta = 0.5 * self.ng.s();
                components
                    .iter()
                    .map(|cpnt| {
                        let rho = cpnt.sigma().recip().powi(2);
                        Gamma::new_unchecked(alpha, beta).ln_f(&rho)
                    })
                    .sum::<f64>()
            };

            let result = mh_prior(
                self.ng.v(),
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
            ln_prior += result.score_x + self.hyper.pr_v.ln_f(&new_v);
        }

        self.ng =
            NormalGamma::new(self.ng.m(), self.ng.r(), self.ng.s(), new_v)
                .unwrap();

        ln_prior
    }
}

/// Hyper-prior for Normal Gamma (`Ng`)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NigHyper {
    // TODO: Change these to the correct distributions, according to the
    // rasmussen IGMM paper
    /// Prior on `m`
    pub pr_m: Gaussian,
    /// Prior on `r`
    pub pr_r: Gamma,
    /// Prior on `s`
    pub pr_s: Gamma,
    /// Prior on `v`
    pub pr_v: Gamma,
}

impl Default for NigHyper {
    fn default() -> Self {
        NigHyper {
            pr_m: Gaussian::new(0.0, 1.0).unwrap(),
            pr_r: Gamma::new(2.0, 2.0).unwrap(),
            pr_s: Gamma::new(2.0, 2.0).unwrap(),
            pr_v: Gamma::new(2.0, 2.0).unwrap(),
        }
    }
}

impl NigHyper {
    pub fn new(pr_m: Gaussian, pr_r: Gamma, pr_s: Gamma, pr_v: Gamma) -> Self {
        NigHyper {
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
        NigHyper {
            pr_m: Gaussian::new(0.0, 0.1).unwrap(),
            pr_r: Gamma::new(40.0, 4.0).unwrap(),
            pr_s: Gamma::new(40.0, 4.0).unwrap(),
            pr_v: Gamma::new(40.0, 4.0).unwrap(),
        }
    }

    /// Vague prior from the data.
    pub fn from_data(xs: &[f64]) -> Self {
        let m = mean(xs);
        let v = var(xs);
        let s = v.sqrt();
        NigHyper {
            pr_m: Gaussian::new(m, s).unwrap(),
            pr_r: Gamma::new(2.0, 1.0).unwrap(),
            pr_s: Gamma::new(s, 1.0 / s).unwrap(),
            pr_v: Gamma::new(2.0, 1.0).unwrap(),
        }
    }

    /// Draw an `Ng` from the hyper
    pub fn draw(&self, mut rng: &mut impl Rng) -> Ng {
        Ng::new(
            self.pr_m.draw(&mut rng),
            self.pr_r.draw(&mut rng),
            self.pr_s.draw(&mut rng),
            self.pr_v.draw(&mut rng),
            self.clone(),
        )
    }
}

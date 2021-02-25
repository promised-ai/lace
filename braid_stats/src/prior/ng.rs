use braid_utils::{mean, var};
use rand::Rng;
use rv::dist::{Gamma, Gaussian, InvGamma, NormalInvGamma};
use rv::traits::*;
use serde::{Deserialize, Serialize};

// use crate::mh::mh_symrw_adaptive_mv;
use crate::mh::mh_prior;
use crate::mh::mh_symrw_adaptive;
use crate::UpdatePrior;

/// Default prior parameters for Geweke testing
pub fn geweke() -> NormalInvGamma {
    NormalInvGamma::new_unchecked(0.0, 1.0, 1.0, 1.0)
}

/// Creates an `Ng` with a vague hyper-prior derived from the data
pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> NormalInvGamma {
    NgHyper::from_data(&xs).draw(&mut rng)
}

/// Draws an `Ng` given a hyper-prior
pub fn from_hyper(hyper: NgHyper, mut rng: &mut impl Rng) -> NormalInvGamma {
    hyper.draw(&mut rng)
}

impl UpdatePrior<f64, Gaussian, NgHyper> for NormalInvGamma {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Gaussian],
        hyper: &NgHyper,
        mut rng: &mut R,
    ) -> f64 {
        let new_m: f64;
        let new_v: f64;
        let new_a: f64;
        let new_b: f64;
        let mut ln_prior = 0.0;

        // TODO: We could get more aggressive with the catching. We could pre-compute the
        // ln(sigma) for each component.
        // struct Gauss {
        //     mu: f64,
        //     sigma: f64,
        //     precision: f64,
        //     ln_sigma: f64,
        // }

        // let gausses: Vec<Gauss> = components
        //     .iter()
        //     .map(|cpnt| {
        //         let sigma = cpnt.sigma();
        //         let ln_sigma = sigma.ln();
        //         let precision = sigma.recip().powi(2);
        //         Gauss {
        //             mu: cpnt.mu(),
        //             ln_sigma,
        //             precision,
        //             sigma,
        //         }
        //     })
        //     .collect();

        // TODO: Can we macro these away?
        {
            let draw = |mut rng: &mut R| hyper.pr_m.draw(&mut rng);
            // let rs = self.r().recip().sqrt();
            // let ln_rs = rs.ln();

            let f = |m: &f64| {
                let ng = NormalInvGamma::new_unchecked(
                    *m,
                    self.v(),
                    self.a(),
                    self.b(),
                );
                components.iter().map(|cpnt| ng.ln_f(cpnt)).sum::<f64>()
                // let errs = gausses
                //     .iter()
                //     .map(|cpnt| {
                //         let sigma = cpnt.sigma * rs;
                //         cpnt.ln_sigma
                //             + ln_rs
                //             + 0.5 * ((m - cpnt.mu) / sigma).powi(2)
                //     })
                //     .sum::<f64>();
                // -errs
            };
            let result = mh_prior(
                self.m(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            // --- Random walk ---
            // let f = |m: f64| {
            //     let errs = gausses
            //         .iter()
            //         .map(|cpnt| {
            //             let sigma = cpnt.sigma * rs;
            //             cpnt.ln_sigma
            //                 + ln_rs
            //                 + 0.5 * ((m - cpnt.mu) / sigma).powi(2)
            //         })
            //         .sum::<f64>();
            //     -errs + hyper.pr_m.ln_f(&m)
            // };

            // // println!("= NG mu");
            // let result = mh_symrw_adaptive(
            //     self.m(),
            //     hyper.pr_m.mu(),
            //     hyper.pr_m.sigma().powi(2) / 10.0,
            //     braid_consts::MH_PRIOR_ITERS,
            //     f,
            //     (std::f64::NEG_INFINITY, std::f64::INFINITY),
            //     &mut rng,
            // );

            new_m = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_m.ln_f(&new_m);
        }

        self.set_m(new_m).unwrap();

        // update v
        {
            let draw = |mut rng: &mut R| hyper.pr_v.draw(&mut rng);
            let f = |v: &f64| {
                let ng = NormalInvGamma::new_unchecked(
                    self.m(),
                    *v,
                    self.a(),
                    self.b(),
                );
                components.iter().map(|cpnt| ng.ln_f(cpnt)).sum::<f64>()
                // let rs = (*r).recip().sqrt();
                // let ln_rs = rs.ln();
                // let m = self.m();
                // let errs = gausses
                //     .iter()
                //     .map(|cpnt| {
                //         let sigma = cpnt.sigma * rs;
                //         cpnt.ln_sigma
                //             + ln_rs
                //             + 0.5 * ((m - cpnt.mu) / sigma).powi(2)
                //     })
                //     .sum::<f64>();
                // -errs
            };

            let result = mh_prior(
                self.v(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );
            //
            // --- Random walk ---
            // let f = |r: f64| {
            //     let rs = r.recip().sqrt();
            //     let ln_rs = rs.ln();
            //     let m = self.m();
            //     let errs = gausses
            //         .iter()
            //         .map(|cpnt| {
            //             let sigma = cpnt.sigma * rs;
            //             cpnt.ln_sigma
            //                 + ln_rs
            //                 + 0.5 * ((m - cpnt.mu) / sigma).powi(2)
            //         })
            //         .sum::<f64>();
            //     -errs + hyper.pr_r.ln_f(&r)
            // };

            // // println!("= NG r");
            // let result = mh_symrw_adaptive(
            //     self.r(),
            //     hyper.pr_r.mean().unwrap_or(1.0),
            //     hyper.pr_r.variance().unwrap_or(10.0) / 100.0,
            //     braid_consts::MH_PRIOR_ITERS,
            //     f,
            //     (0.0, std::f64::INFINITY),
            //     &mut rng,
            // );

            new_v = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_v.ln_f(&new_v);
        }

        self.set_v(new_v).unwrap();

        // update a
        {
            // let shape = 0.5 * self.v();
            let draw = |mut rng: &mut R| hyper.pr_a.draw(&mut rng);
            let f = |a: &f64| {
                let ng = NormalInvGamma::new_unchecked(
                    self.m(),
                    self.v(),
                    *a,
                    self.b(),
                );
                components.iter().map(|cpnt| ng.ln_f(cpnt)).sum::<f64>()
                // we can save a good chunk of time by never computing the
                // gamma(shape) term because we don't need it because we're not
                // re-sampling shape
                // let rate = 0.5 * s;
                // let ln_rate = rate.ln();

                // gausses
                //     .iter()
                //     .map(|cpnt| {
                //         let rho = cpnt.sigma.recip().powi(2);
                //         let ln_rho = -2.0 * cpnt.ln_sigma;
                //         shape * ln_rate + (shape - 1.0) * ln_rho - (rate * rho)
                //     })
                //     .sum::<f64>()
            };
            let result = mh_prior(
                self.a(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            // let f = |s: f64| {
            //     // we can save a good chunk of time by never computing the
            //     // gamma(shape) term because we don't need it because we're not
            //     // re-sampling shape
            //     let rate = 0.5 * s;
            //     let ln_rate = rate.ln();

            //     let sum = gausses
            //         .iter()
            //         .map(|cpnt| {
            //             let rho = cpnt.sigma.recip().powi(2);
            //             let ln_rho = -2.0 * cpnt.ln_sigma;
            //             shape * ln_rate + (shape - 1.0) * ln_rho - (rate * rho)
            //         })
            //         .sum::<f64>();
            //     sum + hyper.pr_s.ln_f(&s)
            // };

            // // println!("= NG s");
            // let result = mh_symrw_adaptive(
            //     self.s(),
            //     hyper.pr_s.mean().unwrap_or(1.0),
            //     hyper.pr_s.variance().unwrap_or(10.0) / 100.0,
            //     braid_consts::MH_PRIOR_ITERS,
            //     f,
            //     (0.0, std::f64::INFINITY),
            //     &mut rng,
            // );

            new_a = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_a.ln_f(&new_a);
        }

        self.set_a(new_a).unwrap();

        // update b
        {
            // use special::Gamma;
            let draw = |mut rng: &mut R| hyper.pr_b.draw(&mut rng);

            // let rate = 0.5 * self.s();
            // let ln_rate = rate.ln();
            let f = |b: &f64| {
                let ng = NormalInvGamma::new_unchecked(
                    self.m(),
                    self.v(),
                    self.a(),
                    *b,
                );
                components.iter().map(|cpnt| ng.ln_f(cpnt)).sum::<f64>()
                // let shape = 0.5 * v;
                // let ln_gamma_shape = shape.ln_gamma().0;
                // gausses
                //     .iter()
                //     .map(|cpnt| {
                //         let rho = cpnt.sigma.recip().powi(2);
                //         let ln_rho = -2.0 * cpnt.ln_sigma;
                //         shape * ln_rate - ln_gamma_shape
                //             + (shape - 1.0) * ln_rho
                //             - (rate * rho)
                //     })
                //     .sum::<f64>()
            };

            let result = mh_prior(
                self.b(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            // let f = |v: f64| {
            //     let shape = 0.5 * v;
            //     let ln_gamma_shape = shape.ln_gamma().0;
            //     let sum = gausses
            //         .iter()
            //         .map(|cpnt| {
            //             let rho = cpnt.sigma.recip().powi(2);
            //             let ln_rho = -2.0 * cpnt.ln_sigma;
            //             shape * ln_rate - ln_gamma_shape
            //                 + (shape - 1.0) * ln_rho
            //                 - (rate * rho)
            //         })
            //         .sum::<f64>();
            //     sum + hyper.pr_v.ln_f(&v)
            // };

            // // println!("= NG v");
            // let result = mh_symrw_adaptive(
            //     self.v(),
            //     hyper.pr_v.mean().unwrap_or(1.0),
            //     hyper.pr_v.variance().unwrap_or(10.0) / 100.0,
            //     braid_consts::MH_PRIOR_ITERS,
            //     f,
            //     (0.0, std::f64::INFINITY),
            //     &mut rng,
            // );

            new_b = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            // FIXME; score_x is invalid now
            ln_prior += result.score_x + hyper.pr_b.ln_f(&new_b);
        }

        self.set_b(new_b).unwrap();

        ln_prior

        // --- BEGIN Symmetric random walk MCMC ---
        // let score_fn = |mvab: &[f64]| {
        //     let m = mvab[0];
        //     let v = mvab[1];
        //     let a = mvab[2];
        //     let b = mvab[3];

        //     let scale = v.sqrt();

        //     let invgam = InvGamma::new(a, b).unwrap();
        //     let loglike = gausses
        //         .iter()
        //         .map(|cpnt| {
        //             let gauss = Gaussian::new(m, scale * cpnt.sigma).unwrap();
        //             gauss.ln_f(&cpnt.mu) + invgam.ln_f(&cpnt.sigma.powi(2))
        //         })
        //         .sum::<f64>();

        //     let prior = hyper.pr_m.ln_f(&m)
        //         + hyper.pr_v.ln_f(&v)
        //         + hyper.pr_a.ln_f(&a)
        //         + hyper.pr_b.ln_f(&b);
        //     loglike + prior
        // };

        // use crate::mat::{Vector4, Matrix4x4};

        // // Variance elements for the random walk proposal
        // let proposal_var_diag: [f64; 4] = [
        //     0.2 * hyper.pr_m.variance().unwrap(),
        //     0.2 * hyper.pr_v.variance().unwrap_or(1.0),
        //     0.2 * hyper.pr_a.variance().unwrap_or(1.0),
        //     0.2 * hyper.pr_b.variance().unwrap_or(1.0),
        // ];

        // let mh_result = mh_symrw_adaptive_mv(
        //     Vector4([self.m(), self.v(), self.a(), self.b()]),
        //     Vector4([
        //         hyper.pr_m.mean().unwrap(),
        //         hyper.pr_v.mean().unwrap_or(1.0),
        //         hyper.pr_a.mean().unwrap_or(1.0),
        //         hyper.pr_b.mean().unwrap_or(1.0),
        //     ]),
        //     Matrix4x4::from_diag(proposal_var_diag),
        //     50,
        //     score_fn,
        //     &vec![
        //         (std::f64::NEG_INFINITY, std::f64::INFINITY),
        //         (0.0, std::f64::INFINITY),
        //         (0.0, std::f64::INFINITY),
        //         (0.0, std::f64::INFINITY),
        //     ],
        //     &mut rng
        // );

        // self.set_m(mh_result.x[0]).unwrap();
        // self.set_v(mh_result.x[1]).unwrap();
        // self.set_a(mh_result.x[2]).unwrap();
        // self.set_b(mh_result.x[3]).unwrap();
        // // self.set_v(4.0).unwrap();

        // hyper.pr_m.ln_f(&self.m())
        //     + hyper.pr_v.ln_f(&self.v())
        //     + hyper.pr_a.ln_f(&self.a())
        //     + hyper.pr_b.ln_f(&self.b())
    }
}

/// Hyper-prior for Normal Gamma (`Ng`)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NgHyper {
    /// Prior on `m`
    pub pr_m: Gaussian,
    /// Prior on `v`
    pub pr_v: InvGamma,
    /// Prior on `a`
    pub pr_a: InvGamma,
    /// Prior on `b`
    pub pr_b: InvGamma,
}

impl Default for NgHyper {
    fn default() -> Self {
        NgHyper {
            pr_m: Gaussian::new_unchecked(0.0, 1.0),
            pr_v: InvGamma::new_unchecked(2.0, 1.0),
            pr_a: InvGamma::new_unchecked(2.0, 1.0),
            pr_b: InvGamma::new_unchecked(2.0, 1.0),
        }
    }
}

impl NgHyper {
    pub fn new(
        pr_m: Gaussian,
        pr_v: InvGamma,
        pr_a: InvGamma,
        pr_b: InvGamma,
    ) -> Self {
        NgHyper {
            pr_m,
            pr_v,
            pr_a,
            pr_b,
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
            pr_v: InvGamma::new(20.0, 40.0).unwrap(),
            pr_a: InvGamma::new(20.0, 40.0).unwrap(),
            pr_b: InvGamma::new(20.0, 40.0).unwrap(),
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
        let sqrt_n = (xs.len() as f64).sqrt();
        let m = mean(xs);
        let v = var(xs);
        let s = v.sqrt();

        NgHyper {
            pr_m: Gaussian::new(m, s).unwrap(),
            pr_v: InvGamma::new(sqrt_n, sqrt_n * s).unwrap(),
            pr_a: InvGamma::new_unchecked(2.0, 1.0),
            pr_b: InvGamma::new(sqrt_n, sqrt_n * s / 10.0).unwrap(),
        }
    }

    /// Draw an `Ng` from the hyper
    pub fn draw(&self, mut rng: &mut impl Rng) -> NormalInvGamma {
        NormalInvGamma::new_unchecked(
            self.pr_m.draw(&mut rng),
            self.pr_v.draw(&mut rng),
            self.pr_a.draw(&mut rng),
            self.pr_b.draw(&mut rng),
        )
    }
}

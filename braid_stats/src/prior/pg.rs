use rand::Rng;
// use rv::data::DataOrSuffStat;
use rv::dist::{Gamma, InvGamma, Poisson};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::mh::mh_prior;
use crate::UpdatePrior;

pub fn geweke() -> Gamma {
    Gamma::new_unchecked(10.0, 10.0)
}

/// Draw the prior from the hyper-prior
pub fn from_hyper(hyper: PgHyper, mut rng: &mut impl Rng) -> Gamma {
    hyper.draw(&mut rng)
}

/// Build a vague hyper-prior given `k` and draws the prior from that
pub fn from_data(xs: &[u32], mut rng: &mut impl Rng) -> Gamma {
    PgHyper::from_data(&xs).draw(&mut rng)
}

// impl ConjugatePrior<u32, Poisson> for Pg {
//     type Posterior = Gamma;
//     type LnMCache = <Gamma as ConjugatePrior<u32, Poisson>>::LnMCache;
//     type LnPpCache = <Gamma as ConjugatePrior<u32, Poisson>>::LnPpCache;

//     #[inline]
//     fn posterior(&self, x: &DataOrSuffStat<u32, Poisson>) -> Gamma {
//         self.gamma.posterior(&x)
//     }

//     #[inline]
//     fn ln_m_cache(&self) -> Self::LnMCache {
//         <Gamma as ConjugatePrior<u32, Poisson>>::ln_m_cache(&self.gamma)
//     }

//     #[inline]
//     fn ln_m_with_cache(
//         &self,
//         cache: &Self::LnMCache,
//         x: &DataOrSuffStat<u32, Poisson>,
//     ) -> f64 {
//         self.gamma.ln_m_with_cache(cache, x)
//     }

//     #[inline]
//     fn ln_pp_cache(&self, x: &DataOrSuffStat<u32, Poisson>) -> Self::LnPpCache {
//         self.gamma.ln_pp_cache(x)
//     }

//     #[inline]
//     fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &u32) -> f64 {
//         self.gamma.ln_pp_with_cache(cache, y)
//     }
// }

impl UpdatePrior<u32, Poisson, PgHyper> for Gamma {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Poisson],
        hyper: &PgHyper,
        mut rng: &mut R,
    ) -> f64 {
        use special::Gamma;

        let new_shape: f64;
        let new_rate: f64;
        let mut ln_prior = 0.0;

        struct Pois {
            rate: f64,
            ln_rate: f64,
        }

        let poissons: Vec<Pois> = components
            .iter()
            .map(|cpnt| Pois {
                rate: cpnt.rate(),
                ln_rate: cpnt.rate().ln(),
            })
            .collect();

        // TODO: Can we macro these away?
        {
            let draw = |mut rng: &mut R| hyper.pr_shape.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let rate = self.rate();
            let ln_rate = rate.ln();
            let f = |shape: &f64| {
                // let h = self.hyper.clone();
                // let pg = Pg::new(*shape, self.gamma.rate(), h);
                // components
                //     .iter()
                //     .fold(0.0, |logf, cpnt| logf + pg.ln_f(&cpnt))
                let ln_gamma_shape = shape.ln_gamma().0;
                poissons
                    .iter()
                    .map(|cpnt| {
                        // gamma.ln_f(&cpnt.rate())
                        let x = cpnt.rate;
                        let ln_x = cpnt.ln_rate;
                        shape * ln_rate - ln_gamma_shape + (shape - 1.0) * ln_x
                            - (rate * x)
                    })
                    .sum::<f64>()
            };

            let result = mh_prior(
                self.shape(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            new_shape = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            ln_prior += result.score_x + hyper.pr_shape.ln_f(&new_shape);
        }

        self.set_shape(new_shape).unwrap();

        {
            let draw = |mut rng: &mut R| hyper.pr_rate.draw(&mut rng);
            let shape = self.shape();
            // TODO: don't clone hyper every time f is called!
            let f = |rate: &f64| {
                // let h = self.hyper.clone();
                // let pg = Pg::new(self.gamma.shape(), *rate, h);
                // components
                //     .iter()
                //     .fold(0.0, |logf, cpnt| logf + pg.ln_f(&cpnt))

                let ln_rate = rate.ln();
                poissons
                    .iter()
                    .map(|cpnt| {
                        let x = cpnt.rate;
                        let ln_x = cpnt.ln_rate;
                        shape * ln_rate + (shape - 1.0) * ln_x - (rate * x)
                    })
                    .sum::<f64>()
            };

            let result = mh_prior(
                self.rate(),
                f,
                draw,
                braid_consts::MH_PRIOR_ITERS,
                &mut rng,
            );

            new_rate = result.x;

            // mh_prior score is the likelihood of the component parameters
            // under the prior. We have to compute the likelihood of the new
            // prior parameters under the hyperprior manually.
            ln_prior += result.score_x + hyper.pr_rate.ln_f(&new_shape);
        }

        self.set_rate(new_rate).unwrap();

        ln_prior
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PgHyper {
    pub pr_shape: InvGamma,
    pub pr_rate: InvGamma,
}

impl Default for PgHyper {
    fn default() -> Self {
        PgHyper {
            pr_shape: InvGamma::new(1.0, 1.0).unwrap(),
            pr_rate: InvGamma::new(1.0, 1.0).unwrap(),
        }
    }
}

impl PgHyper {
    pub fn new(pr_shape: InvGamma, pr_rate: InvGamma) -> Self {
        PgHyper { pr_shape, pr_rate }
    }

    pub fn geweke() -> Self {
        PgHyper {
            pr_shape: InvGamma::new_unchecked(10.0, 10.0),
            pr_rate: InvGamma::new_unchecked(10.0, 10.0),
        }
    }

    pub fn from_data(xs: &[u32]) -> PgHyper {
        let xsf: Vec<f64> = xs.iter().map(|&x| f64::from(x)).collect();

        let nf = xsf.len() as f64;
        let m = xsf.iter().sum::<f64>() / nf;
        let v = xsf.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / nf;

        // Priors chosen so that mean of rate is the mean of the data and that
        // the variance of rate is variance of the data. That is, we want the
        // prior parameters μ = α/β and v = α^2/β
        PgHyper {
            // input validation so we can get a panic if something goes wrong
            pr_shape: InvGamma::new(2.0, v.recip()).unwrap(),
            pr_rate: InvGamma::new(2.0, v / m).unwrap(),
        }
    }

    pub fn draw(&self, mut rng: &mut impl Rng) -> Gamma {
        Gamma::new_unchecked(
            self.pr_shape.draw(&mut rng),
            self.pr_rate.draw(&mut rng),
        )
    }
}

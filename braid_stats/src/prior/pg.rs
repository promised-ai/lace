use rand::Rng;
use rv::dist::{Gamma, InvGamma, Poisson};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::mh::mh_symrw_adaptive_mv;
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
    PgHyper::from_data(xs).draw(&mut rng)
}

impl UpdatePrior<u32, Poisson, PgHyper> for Gamma {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Poisson],
        hyper: &PgHyper,
        mut rng: &mut R,
    ) -> f64 {
        let rates: Vec<f64> =
            components.iter().map(|cpnt| cpnt.rate()).collect();

        let score_fn = |shape_rate: &[f64]| {
            let shape = shape_rate[0];
            let rate = shape_rate[1];
            let gamma = Gamma::new(shape, rate).unwrap();
            let loglike =
                rates.iter().map(|rate| gamma.ln_f(rate)).sum::<f64>();
            let prior = hyper.pr_rate.ln_f(&rate) + hyper.pr_shape.ln_f(&shape);
            loglike + prior
        };

        use crate::mat::{Matrix2x2, Vector2};

        // XXX; This is a janky sampler and might have problems with being
        // symmetric positive definite.
        let mh_result = mh_symrw_adaptive_mv(
            Vector2([self.shape(), self.rate()]),
            Vector2([
                hyper.pr_shape.mean().unwrap_or(1.0),
                hyper.pr_rate.mean().unwrap_or(1.0),
            ]),
            Matrix2x2::from_diag([
                hyper.pr_shape.variance().unwrap_or(1.0),
                hyper.pr_rate.variance().unwrap_or(1.0),
            ]),
            50,
            score_fn,
            &[(0.0, std::f64::INFINITY), (0.0, std::f64::INFINITY)],
            &mut rng,
        );
        self.set_shape(mh_result.x[0]).unwrap();
        self.set_rate(mh_result.x[1]).unwrap();

        hyper.pr_shape.ln_f(&self.shape()) + hyper.pr_rate.ln_f(&self.rate())
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

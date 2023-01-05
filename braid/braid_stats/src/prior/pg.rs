use crate::rv::dist::{Gamma, InvGamma, Poisson};
use crate::rv::traits::*;
use rand::Rng;
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
pub fn from_data(xs: &[u32]) -> Gamma {
    let nf = xs.len() as f64;
    let rate = xs.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
    Gamma::new_unchecked(rate, 1.0)
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
    pub pr_shape: Gamma,
    pub pr_rate: InvGamma,
}

impl Default for PgHyper {
    fn default() -> Self {
        PgHyper {
            pr_shape: Gamma::new(1.0, 1.0).unwrap(),
            pr_rate: InvGamma::new(1.0, 1.0).unwrap(),
        }
    }
}

impl PgHyper {
    pub fn new(pr_shape: Gamma, pr_rate: InvGamma) -> Self {
        PgHyper { pr_shape, pr_rate }
    }

    pub fn geweke() -> Self {
        PgHyper {
            pr_shape: Gamma::new_unchecked(10.0, 10.0),
            pr_rate: InvGamma::new_unchecked(10.0, 10.0),
        }
    }

    pub fn from_data(xs: &[u32]) -> PgHyper {
        // Here we get the ML gamma parameters for xs.
        // https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation
        // Note that we add a buffer to the numbers (0.1) to avoid domain errors
        // with ln.
        let nf = xs.len() as f64;
        let mean_x = xs.iter().map(|&x| f64::from(x) + 0.1).sum::<f64>() / nf;

        let sum_x = xs.iter().sum::<u32>();
        assert_ne!(sum_x, 0, "`xs` is all zeros.");

        let mean_lnx =
            xs.iter().map(|&x| (f64::from(x) + 0.1).ln()).sum::<f64>() / nf;

        let s = mean_x.ln() - mean_lnx;
        let shape = (3.0 - s + (s - 3.0).mul_add(s - 3.0, 24.0 * s).sqrt())
            / (12.0 * s);
        let scale = mean_x / shape;

        assert!(shape > 0.0, "pg hyper: zero or negative shape: {}", shape);
        assert!(scale > 0.0, "pg hyper: zero or negative scale: {}", scale);

        PgHyper {
            pr_shape: Gamma::new_unchecked(shape, 1.0),
            pr_rate: InvGamma::new_unchecked(scale, 1.0),
        }
    }

    pub fn draw(&self, mut rng: &mut impl Rng) -> Gamma {
        Gamma::new_unchecked(
            self.pr_shape.draw(&mut rng),
            self.pr_rate.draw(&mut rng),
        )
    }
}

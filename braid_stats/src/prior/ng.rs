use braid_utils::mean_var;
use rand::Rng;
use rv::dist::{Gamma, Gaussian, NormalInvChiSquared};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::mh::mh_symrw_adaptive_mv;
use crate::UpdatePrior;

/// Default prior parameters for Geweke testing
pub fn geweke() -> NormalInvChiSquared {
    NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0)
}

/// Creates an `Ng` with a vague hyper-prior derived from the data
pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> NormalInvChiSquared {
    NgHyper::from_data(&xs).draw(&mut rng)
}

/// Draws an `Ng` given a hyper-prior
pub fn from_hyper(
    hyper: NgHyper,
    mut rng: &mut impl Rng,
) -> NormalInvChiSquared {
    hyper.draw(&mut rng)
}

impl UpdatePrior<f64, Gaussian, NgHyper> for NormalInvChiSquared {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Gaussian],
        hyper: &NgHyper,
        mut rng: &mut R,
    ) -> f64 {
        // XXX: What can we save time with caching for each component? For
        // example, if the ln_f function calls sigma.ln() every time, we can
        // cache that value instead of re-computing each time score_fn is called
        // let score_fn = |mkvs2: &[f64]| {
        //     let m = mkvs2[0];
        //     let k = mkvs2[1];
        //     let v = mkvs2[2];
        //     let s2 = mkvs2[3];

        //     let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        //     let loglike = components
        //         .iter()
        //         .map(|cpnt| nix.ln_f(cpnt))
        //         .sum::<f64>();

        //     let prior = hyper.pr_m.ln_f(&m)
        //         + hyper.pr_k.ln_f(&k)
        //         + hyper.pr_v.ln_f(&v)
        //         + hyper.pr_s2.ln_f(&s2);

        //     loglike + prior
        // };

        // use crate::mat::{Vector4, Matrix4x4};

        // // Variance elements for the random walk proposal
        // let proposal_var_diag: [f64; 4] = [
        //     0.2 * hyper.pr_m.variance().unwrap(),
        //     0.2 * hyper.pr_k.variance().unwrap_or(1.0),
        //     0.2 * hyper.pr_v.variance().unwrap_or(1.0),
        //     0.2 * hyper.pr_s2.variance().unwrap_or(1.0),
        // ];

        // let mh_result = mh_symrw_adaptive_mv(
        //     Vector4([self.m(), self.k(), self.v(), self.s2()]),
        //     Vector4([
        //         hyper.pr_m.mean().unwrap(),
        //         hyper.pr_k.mean().unwrap_or(1.0),
        //         hyper.pr_v.mean().unwrap_or(1.0),
        //         hyper.pr_s2.mean().unwrap_or(1.0),
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
        // self.set_k(mh_result.x[1]).unwrap();
        // self.set_v(mh_result.x[2]).unwrap();
        // self.set_s2(mh_result.x[3]).unwrap();

        // XXX: What can we save time with caching for each component? For
        // example, if the ln_f function calls sigma.ln() every time, we can
        // cache that value instead of re-computing each time score_fn is called
        let loglike = |nix: &NormalInvChiSquared| {
            components.iter().map(|cpnt| nix.ln_f(cpnt)).sum::<f64>()
        };

        use crate::mh::mh_prior;
        let mh_result = mh_prior(
            self.clone(),
            loglike,
            |mut rng| hyper.draw(&mut rng),
            200,
            rng,
        );

        *self = mh_result.x;

        hyper.pr_m.ln_f(&self.m())
            + hyper.pr_k.ln_f(&self.k())
            + hyper.pr_v.ln_f(&self.v())
            + hyper.pr_s2.ln_f(&self.s2())
    }
}

/// Hyper-prior for Normal Gamma (`Ng`)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NgHyper {
    /// Prior on `m`
    pub pr_m: Gaussian,
    /// Prior on `v`
    pub pr_k: Gamma,
    /// Prior on `a`
    pub pr_v: Gamma,
    /// Prior on `b`
    pub pr_s2: Gamma,
}

impl Default for NgHyper {
    fn default() -> Self {
        NgHyper {
            pr_m: Gaussian::new_unchecked(0.0, 1.0),
            pr_k: Gamma::new_unchecked(2.0, 1.0),
            pr_v: Gamma::new_unchecked(2.0, 1.0),
            pr_s2: Gamma::new_unchecked(2.0, 1.0),
        }
    }
}

impl NgHyper {
    pub fn new(pr_m: Gaussian, pr_k: Gamma, pr_v: Gamma, pr_s2: Gamma) -> Self {
        NgHyper {
            pr_m,
            pr_k,
            pr_v,
            pr_s2,
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
            pr_k: Gamma::new(20.0, 20.0).unwrap(),
            pr_v: Gamma::new(20.0, 20.0).unwrap(),
            pr_s2: Gamma::new(20.0, 20.0).unwrap(),
        }
    }

    /// Vague prior from the data.
    pub fn from_data(xs: &[f64]) -> Self {
        // How the prior is set up:
        // - The expected mean should be the mean of the data
        // - The stddev of the mean should be stddev of the data
        // - The expected sttdev should be the stddev of the data
        // let ln_n = (xs.len() as f64).ln();
        let sqrt_n = (xs.len() as f64).sqrt();
        let (m, v) = mean_var(xs);
        let s = v.sqrt();

        NgHyper {
            pr_m: Gaussian::new(m, s).unwrap(),
            pr_k: Gamma::new(2.0, sqrt_n.recip()).unwrap(),
            pr_v: Gamma::new(2.0, sqrt_n.recip()).unwrap(),
            pr_s2: Gamma::new(s, 1.0).unwrap(),
        }
    }

    /// Draw an `Ng` from the hyper
    pub fn draw(&self, mut rng: &mut impl Rng) -> NormalInvChiSquared {
        NormalInvChiSquared::new_unchecked(
            self.pr_m.draw(&mut rng),
            self.pr_k.draw(&mut rng),
            self.pr_v.draw(&mut rng),
            self.pr_s2.draw(&mut rng),
        )
    }
}

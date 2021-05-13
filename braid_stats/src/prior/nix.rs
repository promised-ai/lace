use braid_utils::mean_var;
use rand::Rng;
use rv::dist::{Gamma, Gaussian, InvGamma, NormalInvChiSquared};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::UpdatePrior;

/// Default prior parameters for Geweke testing
pub fn geweke() -> NormalInvChiSquared {
    NormalInvChiSquared::new_unchecked(0.0, 1.0, 10.0, 1.0)
}

/// Creates an `NormalInvChiSquared` with a vague hyper-prior derived from the
/// data.
pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> NormalInvChiSquared {
    NixHyper::from_data(&xs).draw(&mut rng)
}

/// Draws an `Ng` given a hyper-prior
pub fn from_hyper(
    hyper: NixHyper,
    mut rng: &mut impl Rng,
) -> NormalInvChiSquared {
    hyper.draw(&mut rng)
}

impl UpdatePrior<f64, Gaussian, NixHyper> for NormalInvChiSquared {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&Gaussian],
        hyper: &NixHyper,
        rng: &mut R,
    ) -> f64 {
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

/// Hyper-prior for Normal Inverse Chi-Squared (Nix)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NixHyper {
    /// Prior on `m`
    pub pr_m: Gaussian,
    /// Prior on `v`
    pub pr_k: Gamma,
    /// Prior on `a`
    pub pr_v: InvGamma,
    /// Prior on `b`
    pub pr_s2: InvGamma,
}

impl Default for NixHyper {
    fn default() -> Self {
        NixHyper {
            pr_m: Gaussian::new_unchecked(0.0, 1.0),
            pr_k: Gamma::new_unchecked(2.0, 1.0),
            pr_v: InvGamma::new_unchecked(2.0, 2.0),
            pr_s2: InvGamma::new_unchecked(2.0, 2.0),
        }
    }
}

impl NixHyper {
    pub fn new(
        pr_m: Gaussian,
        pr_k: Gamma,
        pr_v: InvGamma,
        pr_s2: InvGamma,
    ) -> Self {
        NixHyper {
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
        NixHyper {
            pr_m: Gaussian::new(0.0, 0.1).unwrap(),
            pr_k: Gamma::new(40.0, 40.0).unwrap(),
            pr_v: InvGamma::new(21.0, 120.0).unwrap(),
            pr_s2: InvGamma::new(40.0, 40.0).unwrap(),
        }
    }

    /// Vague prior from the data.
    pub fn from_data(xs: &[f64]) -> Self {
        // How the prior is set up:
        // - The expected mean should be the mean of the data
        // - The stddev of the mean should be stddev of the data
        // - The expected sttdev should be the stddev of the data
        // let ln_n = (xs.len() as f64).ln();
        let (m, v) = mean_var(xs);
        let s = v.sqrt();
        let logn = (xs.len() as f64).ln();

        NixHyper {
            pr_m: Gaussian::new(m, s).unwrap(),
            pr_k: Gamma::new(1.0, 1.0).unwrap(),
            pr_v: InvGamma::new(logn, logn).unwrap(),
            pr_s2: InvGamma::new(logn, v).unwrap(),
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

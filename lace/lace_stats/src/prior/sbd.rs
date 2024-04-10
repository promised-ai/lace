use crate::rv::dist::Gamma;
use crate::rv::experimental::stick_breaking::{
    StickBreaking, StickBreakingDiscrete,
};
use crate::rv::traits::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::mh::mh_prior;
use crate::UpdatePrior;

/// Default `Csd` for Geweke testing
pub fn geweke() -> StickBreaking {
    StickBreaking::from_alpha(1.0).unwrap()
}

/// Draw the prior from the hyper-prior
pub fn from_hyper(hyper: SbdHyper, mut rng: &mut impl Rng) -> StickBreaking {
    hyper.draw(&mut rng)
}

/// Build a vague hyper-prior given `k` and draws the prior from that
pub fn vague() -> StickBreaking {
    StickBreaking::from_alpha(1.0).unwrap()
}

impl UpdatePrior<usize, StickBreakingDiscrete, SbdHyper> for StickBreaking {
    fn update_prior<R: Rng>(
        &mut self,
        components: &[&StickBreakingDiscrete],
        hyper: &SbdHyper,
        rng: &mut R,
    ) -> f64 {
        let stick_seqs: Vec<_> = components
            .iter()
            .map(|&cpnt| {
                let n_sticks = cpnt.stick_sequence().num_weights_unstable();
                cpnt.stick_sequence().weights(n_sticks)
            })
            .collect();

        let mh_result = {
            let loglike = |alpha: &f64| {
                let sb = StickBreaking::from_alpha(*alpha).unwrap();
                stick_seqs.iter().map(|seq| sb.ln_f(seq)).sum::<f64>()
            };

            mh_prior(
                self.alpha(),
                loglike,
                |rng| hyper.pr_alpha.draw(rng),
                lace_consts::MH_PRIOR_ITERS,
                rng,
            )
        };

        // self.set_alpha(mh_result.x).unwrap();
        *self = StickBreaking::from_alpha(mh_result.x).unwrap();
        mh_result.score_x + hyper.pr_alpha.ln_f(&mh_result.x)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SbdHyper {
    pub pr_alpha: Gamma,
}

impl Default for SbdHyper {
    fn default() -> Self {
        Self {
            pr_alpha: Gamma::default(),
        }
    }
}

impl SbdHyper {
    pub fn new(shape: f64, rate: f64) -> Self {
        SbdHyper {
            pr_alpha: Gamma::new(shape, rate).unwrap(),
        }
    }

    /// A restrictive prior to confine Geweke.
    ///
    /// Since the geweke test seeks to draw samples from the joint of the prior
    /// and the data, p(x, θ), and since θ is indluenced by the hyper-prior, if
    /// the hyper parameters are not tight, the data can go crazy and cause a
    /// bunch of math errors.
    pub fn geweke() -> Self {
        SbdHyper {
            pr_alpha: Gamma::new(5.0, 5.0).unwrap(),
        }
    }

    pub fn vague() -> Self {
        SbdHyper {
            pr_alpha: Gamma::new(1.0, 1.0).unwrap(),
        }
    }

    /// Draw a `Csd` from the hyper-prior
    pub fn draw(&self, mut rng: &mut impl Rng) -> StickBreaking {
        let alpha = self.pr_alpha.draw(&mut rng);
        StickBreaking::from_alpha(alpha).unwrap()
    }
}

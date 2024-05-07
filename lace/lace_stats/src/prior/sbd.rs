use crate::rv::dist::Gamma;
use crate::rv::experimental::stick_breaking_process::{
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
        // let mut n: f64 = 0.0;
        // let sum_ln_x = components
        //     .iter()
        //     .map(|&cpnt| {
        //         let n_sticks = cpnt.stick_sequence().num_weights_unstable();
        //         let seq = cpnt.stick_sequence().weights(n_sticks);
        //         seq.0
        //             .iter()
        //             .fold((0.0, 1.0), |(sum, rm_mass), &w_i| {
        //                 if rm_mass < 1e-16 {
        //                     (sum, rm_mass)
        //                 } else {
        //                     let x_i = 1.0 - w_i / rm_mass;
        //                     n += 1.0;
        //                     (sum + x_i.ln(), rm_mass - w_i)
        //                 }
        //             })
        //             .0
        //     })
        //     .sum::<f64>();
        // let k = components.len() as f64;
        // // eprintln!("{sum_ln_x}");

        // let shape = n + k * hyper.pr_alpha.shape() - k + 1.0;
        // let rate = k * hyper.pr_alpha.rate() - sum_ln_x;

        // // eprintln!("{shape}, {rate}");

        // let alpha: f64 = Gamma::new(shape, rate).unwrap().draw(rng);
        // self.set_alpha(alpha).unwrap();
        // 0.0

        // -----------
        use crate::rv::dist::Beta;

        let mut stat = crate::rv::data::BetaSuffStat::new();

        components.iter().for_each(|cpnt| {
            let n_sticks = cpnt.stick_sequence().num_weights_unstable();
            let seq = cpnt.stick_sequence().weights(n_sticks);
            let mut rm_mass = 1.0;
            seq.0.iter().for_each(|&w_i| {
                let x_i = w_i / rm_mass;
                if x_i > 0.0 && x_i < 1.0 {
                    rm_mass -= w_i;
                    stat.observe(&x_i);
                }
            });
        });

        let loglike = |alpha: &f64| {
            let upl = Beta::new(1.0, *alpha).unwrap();
            <Beta as HasSuffStat<f64>>::ln_f_stat(&upl, &stat)
        };

        let mh_result = mh_prior(
            self.alpha(),
            loglike,
            |rng| hyper.pr_alpha.draw(rng),
            100,
            rng,
        );
        self.set_alpha(mh_result.x).unwrap();
        mh_result.score_x
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

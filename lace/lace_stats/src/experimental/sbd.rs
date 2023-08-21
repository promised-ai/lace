use crate::rv::experimental::Sb;
use crate::rv::traits::Rv;
use crate::{rv::dist::Gamma, UpdatePrior};
use lace_consts::rv::experimental::Sbd;
use lace_consts::rv::prelude::GammaError;
use lace_data::AccumScore;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SbdHyper {
    pub gamma: Gamma,
}

impl Default for SbdHyper {
    fn default() -> Self {
        Self {
            gamma: Gamma::new_unchecked(2.0, 2.0),
        }
    }
}

impl SbdHyper {
    pub fn new(shape: f64, rate: f64) -> Result<Self, GammaError> {
        Ok(Self {
            gamma: Gamma::new(shape, rate)?,
        })
    }

    pub fn draw<R: rand::Rng>(&self, k: usize, rng: &mut R) -> Sb {
        let seed: u64 = rng.gen();
        let alpha = self.gamma.draw(rng);
        Sb::new(alpha, k, Some(seed))
    }
}

impl UpdatePrior<usize, Sbd, SbdHyper> for Sb {
    fn update_prior<R: rand::Rng>(
        &mut self,
        components: &[&Sbd],
        hyper: &SbdHyper,
        rng: &mut R,
    ) -> f64 {
        let ln_prior = |alpha: &f64| {
            let sb = Sb::new(*alpha, 1, None);
            components.iter().map(|cpnt| sb.ln_f(cpnt)).sum::<f64>()
        };

        let res = crate::mh::mh_prior(
            self.alpha(),
            ln_prior,
            |r| hyper.gamma.draw(r),
            200,
            rng,
        );

        self.set_alpha_unchecked(res.x);

        res.score_x
    }
}

use braid_stats::mh::mh_prior;
use braid_stats::UpdatePrior;
use rand::{FromEntropy, Rng};
use rand_xoshiro::Xoshiro256Plus;
use rv::data::DataOrSuffStat;
use rv::dist::Beta;
use rv::traits::{ConjugatePrior, Rv, SuffStat};
use serde::{Deserialize, Serialize};

use crate::integrate::mc_integral;
use crate::labeler::{Label, Labeler, LabelerSuffStat};

#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct LabelerPrior {
    pr_k: Beta,
    pr_h: Beta,
    pr_world: Beta,
}

impl Rv<Labeler> for LabelerPrior {
    fn ln_f(&self, x: &Labeler) -> f64 {
        self.pr_k.ln_f(&x.p_k())
            + self.pr_h.ln_f(&x.p_h())
            + self.pr_world.ln_f(&x.p_world())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Labeler {
        Labeler {
            p_h: self.pr_h.draw(&mut rng),
            p_k: self.pr_k.draw(&mut rng),
            p_world: self.pr_world.draw(&mut rng),
        }
    }
}

impl ConjugatePrior<Label, Labeler> for LabelerPrior {
    // TODO: non-static lifetime
    type Posterior = LabelerPosterior;

    fn posterior(&self, x: &DataOrSuffStat<Label, Labeler>) -> Self::Posterior {
        // TODO: should return hacky function that uses MCMC to draw from the
        // posterior, but raises a runtime error if `f` or `ln_f` is called.
        // TODO: Too much cloning
        let stat = match x {
            DataOrSuffStat::SuffStat(stat) => (*stat).clone(),
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                stat
            }
            DataOrSuffStat::None => unreachable!(),
        };
        LabelerPosterior {
            prior: self.clone(),
            stat: stat,
            n_mh_iters: 100,
        }
    }

    fn ln_m(&self, x: &DataOrSuffStat<Label, Labeler>) -> f64 {
        // XXX: This destroys RNG seed control
        let mut rng = Xoshiro256Plus::from_entropy();
        match x {
            DataOrSuffStat::None => 1.0,
            DataOrSuffStat::SuffStat(stat) => mc_integral(
                |labler| sf_loglike(&stat, &labler),
                |mut r| self.draw(&mut r),
                10_000,
                &mut rng,
            ),
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                mc_integral(
                    |labler| sf_loglike(&stat, &labler),
                    |mut r| self.draw(&mut r),
                    10_000,
                    &mut rng,
                )
            }
        }
    }

    fn ln_pp(&self, y: &Label, x: &DataOrSuffStat<Label, Labeler>) -> f64 {
        unimplemented!();
    }
}

impl UpdatePrior<Label, Labeler> for LabelerPrior {
    fn update_prior<R: Rng>(
        &mut self,
        _components: &Vec<&Labeler>,
        _rng: &mut R,
    ) {
    }
}

// computed log likelihood from the sufficient statistic
fn sf_loglike(xs: &LabelerSuffStat, labeler: &Labeler) -> f64 {
    let mut logp = 0.0;

    if xs.n_truth_tt > 0 {
        logp +=
            xs.n_truth_tt as f64 * labeler.ln_f(&Label::new(true, Some(true)));
    }

    if xs.n_truth_tf > 0 {
        logp +=
            xs.n_truth_tf as f64 * labeler.ln_f(&Label::new(true, Some(false)));
    }

    if xs.n_truth_ft > 0 {
        logp +=
            xs.n_truth_ft as f64 * labeler.ln_f(&Label::new(false, Some(true)));
    }

    if xs.n_truth_ff > 0 {
        logp += xs.n_truth_ff as f64
            * labeler.ln_f(&Label::new(false, Some(false)));
    }

    if xs.n_unk_t > 0 {
        logp += xs.n_unk_t as f64 * labeler.ln_f(&Label::new(true, None));
    }

    if xs.n_unk_f > 0 {
        logp += xs.n_unk_f as f64 * labeler.ln_f(&Label::new(false, None));
    }

    logp
}

pub struct LabelerPosterior {
    prior: LabelerPrior,
    stat: LabelerSuffStat,
    n_mh_iters: usize,
}

impl Rv<Labeler> for LabelerPosterior {
    fn ln_f(&self, labeler: &Labeler) -> f64 {
        let loglike = sf_loglike(&self.stat, &labeler);
        let prior = self.prior.ln_f(&labeler);
        prior + loglike
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Labeler {
        // TODO: This is a crappy way to do this
        mh_prior(
            self.prior.draw(&mut rng),
            |x| self.ln_f(&x),
            |mut r| self.prior.draw(&mut r),
            self.n_mh_iters,
            &mut rng,
        )
    }
}

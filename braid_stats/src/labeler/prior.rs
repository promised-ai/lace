use crate::labeler::{Label, Labeler, LabelerSuffStat};
use crate::mh::mh_prior;
use crate::seq::HaltonSeq;
use crate::UpdatePrior;
use braid_utils::misc::logsumexp;
use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::dist::Kumaraswamy;
use rv::traits::{ConjugatePrior, Rv, SuffStat};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct LabelerPrior {
    pub pr_k: Kumaraswamy,
    pub pr_h: Kumaraswamy,
    pub pr_world: Kumaraswamy,
}

impl Default for LabelerPrior {
    fn default() -> Self {
        LabelerPrior {
            pr_k: Kumaraswamy::new(5.0, 1.0).unwrap(),
            pr_h: Kumaraswamy::new(5.0, 1.0).unwrap(),
            // bowl-shape prior with CDF(0.5) = 0.5
            pr_world: Kumaraswamy::new(0.5, 0.564476).unwrap(),
        }
    }
}

impl Rv<Labeler> for LabelerPrior {
    fn ln_f(&self, x: &Labeler) -> f64 {
        self.pr_k.ln_f(&x.p_k())
            + self.pr_h.ln_f(&x.p_h())
            + self.pr_world.ln_f(&x.p_world())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Labeler {
        let p_h = self.pr_h.draw(&mut rng);
        let p_k = self.pr_k.draw(&mut rng);
        let p_world = self.pr_world.draw(&mut rng);

        Labeler { p_h, p_k, p_world }
    }
}

// Use quasi-monte carlo (QMC) to approximate
fn ln_m(prior: &LabelerPrior, stat: &LabelerSuffStat, n: usize) -> f64 {
    // String together 3 Halton sequences
    let loglikes: Vec<f64> = HaltonSeq::new(2)
        .zip(HaltonSeq::new(3))
        .zip(HaltonSeq::new(5))
        .take(n)
        .map(|((a, b), c)| {
            let labeler = Labeler::new(a, b, c);
            sf_loglike(&stat, &labeler) * prior.f(&labeler)
        })
        .collect();

    logsumexp(&loglikes) - (n as f64).ln()
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
        match x {
            DataOrSuffStat::SuffStat(stat) => ln_m(&self, stat, 1_000),
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                ln_m(&self, &stat, 1_000)
            }
            DataOrSuffStat::None => 1.0,
        }
    }

    fn ln_pp(&self, _y: &Label, _x: &DataOrSuffStat<Label, Labeler>) -> f64 {
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
pub fn sf_loglike(xs: &LabelerSuffStat, labeler: &Labeler) -> f64 {
    xs.n_truth_tt as f64 * labeler.ln_f(&Label::new(true, Some(true)))
        + xs.n_truth_tf as f64 * labeler.ln_f(&Label::new(true, Some(false)))
        + xs.n_truth_ft as f64 * labeler.ln_f(&Label::new(false, Some(true)))
        + xs.n_truth_ff as f64 * labeler.ln_f(&Label::new(false, Some(false)))
        + xs.n_unk_t as f64 * labeler.ln_f(&Label::new(true, None))
        + xs.n_unk_f as f64 * labeler.ln_f(&Label::new(false, None))
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn labeler_prior_should_never_return_1_for_p_world() {
        let mut rng = rand::thread_rng();
        let pr = LabelerPrior::default();

        let none_are_1 = pr
            .sample(100_000, &mut rng)
            .drain(..)
            .all(|x| x.p_world < 1.0);

        assert!(none_are_1);
    }
}

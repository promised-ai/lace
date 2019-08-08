use crate::labeler::{Label, Labeler, LabelerSuffStat};
use crate::mh::mh_prior;
use crate::seq::SobolSeq;
use crate::simplex::SimplexPoint;
use crate::UpdatePrior;
use braid_utils::misc::logsumexp;
use rand::{FromEntropy, Rng};
use rv::data::DataOrSuffStat;
use rv::dist::{Kumaraswamy, SymmetricDirichlet};
use rv::traits::{ConjugatePrior, Rv, SuffStat};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct LabelerPrior {
    pub pr_k: Kumaraswamy,
    pub pr_h: Kumaraswamy,
    pub pr_world: SymmetricDirichlet,
}

impl LabelerPrior {
    pub fn standard(n_labels: u8) -> LabelerPrior {
        LabelerPrior {
            pr_k: Kumaraswamy::new(10.0, 1.0).unwrap(),
            pr_h: Kumaraswamy::new(10.0, 1.0).unwrap(),
            pr_world: SymmetricDirichlet::jeffreys(n_labels.into()).unwrap(),
        }
    }

    pub fn uniform(n_labels: u8) -> LabelerPrior {
        LabelerPrior {
            pr_k: Kumaraswamy::uniform(),
            pr_h: Kumaraswamy::uniform(),
            pr_world: SymmetricDirichlet::new(1.0, n_labels.into()).unwrap(),
        }
    }
}

impl Rv<Labeler> for LabelerPrior {
    fn ln_f(&self, x: &Labeler) -> f64 {
        self.pr_k.ln_f(&x.p_k())
            + self.pr_h.ln_f(&x.p_h())
            + self.pr_world.ln_f(x.p_world().point())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Labeler {
        let p_h = self.pr_h.draw(&mut rng);
        let p_k = self.pr_k.draw(&mut rng);
        let p_world = SimplexPoint::new_unchecked(self.pr_world.draw(&mut rng));

        Labeler { p_h, p_k, p_world }
    }
}

fn ln_m(prior: &LabelerPrior, stat: &LabelerSuffStat, n: usize) -> f64 {
    // // Use quasi-monte carlo (QMC) to approximate
    // // String together 3 Halton sequences
    // let loglikes: Vec<f64> = SobolSeq::new(prior.pr_world.k() + 2)
    //     .take(n)
    //     .map(|mut ps| {
    //         let p_k = ps.pop().unwrap();
    //         let p_h = ps.pop().unwrap();
    //         let p_world = uvec_to_simplex(ps);
    //         let labeler = Labeler::new(p_k, p_h, p_world);
    //         sf_loglike(&stat, &labeler) * prior.f(&labeler)
    //     })
    //     .collect();

    // importance sampling using uniform
    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let q = LabelerPrior::uniform(prior.pr_world.k() as u8);
    let loglikes: Vec<f64> = (0..n)
        .map(|_| {
            let labeler: Labeler = q.draw(&mut rng);
            sf_loglike(&stat, &labeler) + prior.ln_f(&labeler)
                - q.ln_f(&labeler)
        })
        .collect();

    logsumexp(&loglikes) - (n as f64).ln()
}

impl ConjugatePrior<Label, Labeler> for LabelerPrior {
    type Posterior = LabelerPosterior;

    fn posterior(&self, x: &DataOrSuffStat<Label, Labeler>) -> Self::Posterior {
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
            n_mh_iters: 200,
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

    fn ln_pp(&self, y: &Label, x: &DataOrSuffStat<Label, Labeler>) -> f64 {
        // TODO: this is so slow it makes me want to drink draino.
        let mut x_stat = LabelerSuffStat::new();
        x_stat.observe(y);
        match x {
            DataOrSuffStat::SuffStat(stat) => {
                let denom = ln_m(&self, &x_stat, 1_000);
                let mut top_stat = (*stat).clone();
                top_stat.observe(y);
                let numer = ln_m(&self, &top_stat, 1_000);
                numer - denom
            }
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                stat.observe(y);
                let numer = ln_m(&self, &stat, 1_000);
                let denom = ln_m(&self, &x_stat, 1_000);
                numer - denom
            }
            DataOrSuffStat::None => 1.0,
        }
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
    xs.counter.iter().fold(0.0, |sum, (x, &count)| {
        sum + labeler.ln_f(x) * f64::from(count)
    })
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
        let dir =
            SymmetricDirichlet::new(1.0, self.prior.pr_world.k()).unwrap();
        // TODO: This is a crappy way to do this, but it seems to work better
        // than symmetric random walk
        // XXX: When using uniform priors on ph and pk, and symmetric Dirichlet
        // with alpha = 1, we dont need to worry about the transition
        // probability because it is always the same.
        mh_prior(
            self.prior.draw(&mut rng),
            |x| sf_loglike(&self.stat, &x) + self.prior.ln_f(&x),
            |r| {
                let p_world = SimplexPoint::new_unchecked(dir.draw(r));
                Labeler::new(r.gen(), r.gen(), p_world)
            },
            self.n_mh_iters,
            &mut rng,
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // FIXME
}

use crate::integrate::importance_integral;
use crate::labeler::{Label, Labeler, LabelerSuffStat};
use crate::mh::mh_importance;
use crate::simplex::SimplexPoint;
use crate::UpdatePrior;
use rand::{Rng, SeedableRng};
use rv::data::DataOrSuffStat;
use rv::dist::{Kumaraswamy, SymmetricDirichlet};
use rv::traits::{ConjugatePrior, Rv, SuffStat};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

    pub fn importance(n_labels: u8) -> LabelerPrior {
        LabelerPrior {
            pr_k: Kumaraswamy::uniform(),
            pr_h: Kumaraswamy::uniform(),
            pr_world: SymmetricDirichlet::new(0.5, n_labels.into()).unwrap(),
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

    // importance sampling
    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let q = LabelerPrior::importance(prior.pr_world.k() as u8);
    importance_integral(
        |x| sf_loglike(&stat, x) + prior.ln_f(x),
        |mut r| q.draw(&mut r),
        |x| q.ln_f(x),
        n,
        &mut rng,
    )
}

impl ConjugatePrior<Label, Labeler> for LabelerPrior {
    type Posterior = LabelerPosterior;
    type LnMCache = ();
    type LnPpCache = LabelerSuffStat;

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
            stat,
            n_mh_iters: 500,
        }
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::LnMCache {
        ()
    }

    fn ln_m_with_cache(
        &self,
        _cache: &Self::LnMCache,
        x: &DataOrSuffStat<Label, Labeler>,
    ) -> f64 {
        match x {
            DataOrSuffStat::SuffStat(stat) => ln_m(&self, stat, 10_000),
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                ln_m(&self, &stat, 10_000)
            }
            DataOrSuffStat::None => 1.0,
        }
    }

    #[inline]
    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<Label, Labeler>,
    ) -> Self::LnPpCache {
        match x {
            DataOrSuffStat::SuffStat(stat) => (*stat).clone(),
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                stat
            }
            DataOrSuffStat::None => LabelerSuffStat::new(),
        }
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &Label) -> f64 {
        let mut x_stat = LabelerSuffStat::new();
        x_stat.observe(y);
        let mut top_stat = (*cache).clone();

        if top_stat.n() == 0 {
            1.0
        } else {
            let denom = ln_m(&self, &x_stat, 10_000);
            top_stat.observe(y);
            let numer = ln_m(&self, &top_stat, 10_000);
            numer - denom
        }
    }

    fn ln_pp(&self, y: &Label, x: &DataOrSuffStat<Label, Labeler>) -> f64 {
        // TODO: this is so slow it makes me want to drink draino.
        let mut x_stat = LabelerSuffStat::new();
        x_stat.observe(y);
        match x {
            DataOrSuffStat::SuffStat(stat) => {
                let denom = ln_m(&self, &x_stat, 10_000);
                let mut top_stat = (*stat).clone();
                top_stat.observe(y);
                let numer = ln_m(&self, &top_stat, 10_000);
                numer - denom
            }
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = LabelerSuffStat::new();
                stat.observe_many(&xs);
                stat.observe(y);
                let numer = ln_m(&self, &stat, 10_000);
                let denom = ln_m(&self, &x_stat, 10_000);
                numer - denom
            }
            DataOrSuffStat::None => 1.0,
        }
    }
}

impl UpdatePrior<Label, Labeler, ()> for LabelerPrior {
    fn update_prior<R: Rng>(
        &mut self,
        _components: &[&Labeler],
        _hyper: &(),
        _rng: &mut R,
    ) -> f64 {
        0.0
    }
}

// computed log likelihood from the sufficient statistic
pub fn sf_loglike(xs: &LabelerSuffStat, labeler: &Labeler) -> f64 {
    xs.counter.iter().fold(0.0, |sum, (x, &count)| {
        labeler.ln_f(x).mul_add(f64::from(count), sum)
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
        let q = LabelerPrior::importance(self.prior.pr_world.k() as u8);

        // TODO: This is a crappy way to do this, but it seems to work better
        // than symmetric random walk
        mh_importance(
            self.prior.draw(&mut rng),
            |x| sf_loglike(&self.stat, &x) + self.prior.ln_f(&x),
            |r| q.draw(r),
            |x| q.ln_f(x),
            self.n_mh_iters,
            &mut rng,
        )
        .x
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // FIXME: write tests
}

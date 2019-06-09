use braid_stats::UpdatePrior;
use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::dist::Beta;
use rv::traits::{ConjugatePrior, Rv};
use serde::{Deserialize, Serialize};

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
    type Posterior = LabelerPrior;

    fn posterior(&self, x: &DataOrSuffStat<Label, Labeler>) -> Self::Posterior {
        unimplemented!();
    }

    fn ln_m(&self, x: &DataOrSuffStat<Label, Labeler>) -> f64 {
        // let suffstat = match x {
        //     DataOrSuffStat::SuffStat(ref stat) => stat,
        //     DataOrSuffStat::Data(ref xs) => {
        //         let stat = LabelerSuffStat::new();
        //         stat.observe_many(&xs);
        //         &stat
        //     },
        //     DataOrSuffStat::None => &LabelerSuffStat::new(),
        // };
        unimplemented!();
    }

    fn ln_pp(&self, y: &Label, x: &DataOrSuffStat<Label, Labeler>) -> f64 {
        unimplemented!();
    }
}

impl UpdatePrior<Label, Labeler> for LabelerPrior {
    fn update_prior<R: Rng>(
        &mut self,
        components: &Vec<&Labeler>,
        rng: &mut R,
    ) {
        unimplemented!();
    }
}

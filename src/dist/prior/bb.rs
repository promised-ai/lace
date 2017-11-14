extern crate rand;

use self::rand::Rng;
use dist::Bernoulli;


// Beta prior for bernoulli likelihood
// -----------------------------------
struct BetaBernoulli {
    pub a: f64,
    pub b: f64,
}


impl Prior<bool, Bernoulli> for BetaBernoulli {
    fn posterior_draw(&self, data: &Vec<bool>, mut rng: &mut Rng) -> Bernoulli {
        unimplemented!();
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Bernoulli {
        unimplemented!();
    }

    fn marginal_score(&self, y: &Vec<bool>) -> f64 {
        unimplemented!();
    }

    fn update_params(&mut self, components: &Vec<Bernoulli>) {
        unimplemented!();
    }
}

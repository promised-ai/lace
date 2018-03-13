extern crate rand;

use self::rand::Rng;
use dist::Bernoulli;
use dist::prior::Prior;

// Beta prior for bernoulli likelihood
// -----------------------------------
pub struct BetaBernoulli {
    pub a: f64,
    pub b: f64,
}

impl Prior<bool, Bernoulli> for BetaBernoulli {
    fn loglike(&self, _model: &Bernoulli) -> f64 {
        unimplemented!();
    }

    fn posterior_draw(&self, _data: &[bool], mut _rng: &mut Rng) -> Bernoulli {
        unimplemented!();
    }

    fn prior_draw(&self, mut _rng: &mut Rng) -> Bernoulli {
        unimplemented!();
    }

    fn marginal_score(&self, _y: &[bool]) -> f64 {
        unimplemented!();
    }

    fn update_params(&mut self, _components: &[Bernoulli], _rng: &mut Rng) {
        unimplemented!();
    }
}

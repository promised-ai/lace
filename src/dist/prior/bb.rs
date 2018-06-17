extern crate rand;

use self::rand::Rng;
use dist::prior::Prior;
use dist::Bernoulli;

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

    fn posterior_draw(
        &self,
        _data: &[bool],
        mut _rng: &mut impl Rng,
    ) -> Bernoulli {
        unimplemented!();
    }

    fn prior_draw(&self, mut _rng: &mut impl Rng) -> Bernoulli {
        unimplemented!();
    }

    fn marginal_score(&self, _y: &[bool]) -> f64 {
        unimplemented!();
    }

    fn update_params<R: Rng>(
        &mut self,
        _components: &[Bernoulli],
        _rng: &mut R,
    ) {
        unimplemented!();
    }

    fn predictive_score(&self, _x: &bool, _y: &[bool]) -> f64 {
        unimplemented!();
    }
}

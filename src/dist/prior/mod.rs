extern crate rand;

pub mod bb;
pub mod csd;
pub mod nig;

use self::rand::Rng;
pub use dist::prior::bb::BetaBernoulli;
pub use dist::prior::csd::CatSymDirichlet;
pub use dist::prior::nig::NormalInverseGamma;
use dist::traits::Distribution;

// TODO: rename file to priors
pub trait Prior<T, M>
where
    M: Distribution<T>,
{
    fn loglike(&self, model: &M) -> f64;
    fn posterior_draw(&self, data: &[T], rng: &mut Rng) -> M;
    fn prior_draw(&self, rng: &mut Rng) -> M;
    fn marginal_score(&self, data: &[T]) -> f64;
    fn update_params(&mut self, components: &[M], rng: &mut Rng);

    fn draw(&self, data_opt: Option<&Vec<T>>, mut rng: &mut Rng) -> M {
        match data_opt {
            Some(data) => self.posterior_draw(data, &mut rng),
            None => self.prior_draw(&mut rng),
        }
    }

    // Not needed until split-merge or Gibbs implemented:
    fn predictive_score(&self, _x: &T, _y: &[T]) -> f64 {
        unimplemented!();
    }

    fn singleton_score(&self, _y: &T) -> f64 {
        unimplemented!();
    }
}
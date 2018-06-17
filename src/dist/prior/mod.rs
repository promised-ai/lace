extern crate rand;

pub mod bb;
pub mod csd;
pub mod nig;

use self::rand::Rng;
pub use dist::prior::bb::BetaBernoulli;
pub use dist::prior::csd::CatSymDirichlet;
pub use dist::prior::nig::NormalInverseGamma;
use dist::traits::Distribution;

// TODO: Instead of passing slices of values to marginal score, posterior draw,
// and predictive score, might there be a way to incorporate the HasSuffstats
// trait to make these functions more efficient? It's awfully wastefull to
// continually have to partition the data, feed it to the prior, and recompute
// the sufficient statistics.
pub trait Prior<T, M>
where
    T: Clone,
    M: Distribution<T>,
{
    /// Log likleihood (PDF/PMF) of a model under the `Prior`
    fn loglike(&self, model: &M) -> f64;

    /// Draw a model given the `Prior` and the evidence in `data`
    fn posterior_draw(&self, data: &[T], rng: &mut impl Rng) -> M;

    /// Draw a model from the `Prior`
    fn prior_draw(&self, rng: &mut impl Rng) -> M;

    /// Marginal liklihood of the `data` given the `Prior`
    fn marginal_score(&self, data: &[T]) -> f64;

    /// Draw new prior parameters given a set of existing models and the hyper
    /// prior.
    fn update_params<R: Rng>(&mut self, components: &[M], rng: &mut R);

    /// Predictive likelihood of observing `x` given observation `y` and the
    /// `Prior`.
    fn predictive_score(&self, x: &T, y: &[T]) -> f64;

    /// Probability of observing `y` given the `Prior`
    fn singleton_score(&self, y: &T) -> f64 {
        self.marginal_score(&vec![(*y).clone()].as_slice())
    }

    /// Draw a new model from the prior if `data_opt` is `None`, draw from the
    /// posterior otherwise.
    fn draw(&self, data_opt: Option<&Vec<T>>, mut rng: &mut impl Rng) -> M {
        match data_opt {
            Some(data) => self.posterior_draw(data, &mut rng),
            None => self.prior_draw(&mut rng),
        }
    }
}

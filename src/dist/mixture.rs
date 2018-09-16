extern crate rand;
extern crate rv;

use std::marker::PhantomData;

use self::rand::Rng;

use self::rv::dist::Mixture;
use self::rv::dist::{Categorical, Gaussian};
use self::rv::misc::pflip;
use self::rv::traits::*;
use misc::logsumexp;
use optimize::fmin_bounded;

pub fn flat_mixture<Fx>(components: Vec<Fx>) -> Mixture<Fx> {
    let k = components.len();
    let w = (k as f64).recip();
    let weights: Vec<f64> = vec![w; k];
    Mixture::new(weights, components).unwrap()
}

/// Entropy of the mixture estimated via Monte Carlo integration
pub fn entropy_mc<X, Fx, R>(
    mixture: &Mixture<Fx>,
    n: usize,
    mut rng: &mut R,
) -> f64
where
    Fx: Rv<X> + Entropy,
    R: Rng,
{
    let xs: Vec<X> = mixture.sample(n, &mut rng);
    let log_n = (n as f64).ln();
    xs.iter().fold(0.0, |acc, x| {
        let ln_f = mixture.ln_f(x);
        acc - ln_f.exp() * ln_f
    })
}

/// Jensen-Shannon Divergence between all the mixture components estimated via
/// Monte Carlo integration
pub fn jsd_mc<X, Fx, R>(mixture: &Mixture<Fx>, n: usize, mut rng: &mut R) -> f64
where
    Fx: Rv<X> + Entropy,
    R: Rng,
{
    let ln_weights: Vec<f64> =
        mixture.weights.iter().map(|&w| w.ln()).collect();
    let h_all = entropy_mc(&mixture, n, &mut rng);
    let h_sum = mixture
        .components
        .iter()
        .zip(ln_weights)
        .fold(0.0, |acc, (cpnt, lnw)| acc + lnw + cpnt.entropy());

    let kf = mixture.weights.len() as f64;

    (h_all - h_sum) / kf.ln()
}
// TODO: Add tests

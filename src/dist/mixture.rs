// FIXME: Replace this with the rv Mixture
// This will require implementing entropy and JS divergence in mixture.
// I propose implementing them as functions, eg:

// fn entropy<Fx>(m: &Mixture<Fx>, n_samples: usize, rng: &mut impl Rng) -> f64
// fn js<Fx>(m: &Mixture<Fx>, n_samples: usize, rng: &mut impl Rng) -> f64
//
// That way we don't have to wait for them to show up in rv
extern crate rand;
extern crate rv;

use std::marker::PhantomData;

use self::rand::Rng;

use self::rv::dist::{Categorical, Gaussian};
use self::rv::misc::pflip;
use self::rv::traits::*;
use misc::logsumexp;
use optimize::fmin_bounded;

pub struct MixtureModel<X, Fx>
where
    Fx: Rv<X> + Entropy,
{
    components: Vec<Fx>,
    weights: Vec<f64>,
    _phantom: PhantomData<X>,
}

impl<X, Fx> MixtureModel<X, Fx>
where
    Fx: Rv<X> + Entropy,
{
    pub fn flat(components: Vec<Fx>) -> Self {
        let k = components.len();
        let weights = vec![1.0 / (k as f64); k];

        MixtureModel {
            components: components,
            weights: weights,
            _phantom: PhantomData,
        }
    }

    pub fn log_weights(&self) -> Vec<f64> {
        self.weights.iter().map(|w| w.ln()).collect()
    }

    pub fn sample(&self, n: usize, mut rng: &mut impl Rng) -> Vec<X> {
        let mut xs = Vec::with_capacity(n);
        for k in pflip(&self.weights, n, &mut rng) {
            xs.push(self.components[k].draw(&mut rng));
        }
        xs
    }

    pub fn loglike(&self, x: &X) -> f64 {
        let cpnt_loglikes: Vec<f64> = self
            .components
            .iter()
            .zip(&self.weights)
            .map(|(cpnt, w)| w.ln() + cpnt.ln_f(x))
            .collect();

        logsumexp(&cpnt_loglikes)
    }

    pub fn loglikes(&self, xs: &[X]) -> Vec<f64> {
        xs.iter().map(|x| self.loglike(x)).collect()
    }

    pub fn entropy(&self, n_samples: usize, mut rng: &mut impl Rng) -> f64 {
        let xs = self.sample(n_samples, &mut rng);
        let logn = (n_samples as f64).ln();
        self.loglikes(&xs).iter().fold(0.0, |acc, ll| acc + ll) - logn
    }

    /// Normalized Jensen-Shannon divergence
    pub fn js_divergence(
        &self,
        n_samples: usize,
        mut rng: &mut impl Rng,
    ) -> f64 {
        let log_weights = self.log_weights();
        let h_all = self.entropy(n_samples, &mut rng);
        let h_sum = self
            .components
            .iter()
            .zip(log_weights)
            .fold(0.0, |acc, (cpnt, logw)| acc + logw + cpnt.entropy());

        let k = self.weights.len() as f64;

        (h_all - h_sum) / k.ln()
    }
}

impl Mode<f64> for MixtureModel<f64, Gaussian> {
    fn mode(&self) -> Option<f64> {
        let k = self.components.len();
        if k == 1 {
            Some(self.components[0].mu)
        } else {
            let _means: Vec<f64> =
                self.components.iter().map(|cpnt| cpnt.mu).collect();
            let (m0, means) = _means.split_first().unwrap();
            let a = means
                .iter()
                .fold(m0, |min, x| if x < min { x } else { min });
            let b = means
                .iter()
                .fold(m0, |max, x| if x > max { x } else { max });

            Some(fmin_bounded(
                |x| -self.loglike(&x),
                (*a, *b),
                Some(10E-8),
                None,
            ))
        }
    }
}

// FIXME: make generic to unisgned types
impl Mode<u8> for MixtureModel<u8, Categorical> {
    fn mode(&self) -> Option<u8> {
        let k = self.components[0].ln_weights.len();
        let pairs: Vec<(u8, f64)> = (0..k)
            .map(|x| {
                let xi = x as u8;
                (xi, self.loglike(&xi))
            }).collect();

        let (first, rest) = pairs.split_first().unwrap();

        let x = rest
            .iter()
            .fold(
                first,
                |current, nxt| {
                    if nxt.1 > current.1 {
                        nxt
                    } else {
                        current
                    }
                },
            ).0;
        Some(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn new_gaussian() {
        let g1 = Gaussian::new(0.0, 1.0).unwrap();
        let g2 = Gaussian::new(2.0, 3.0).unwrap();

        let m = MixtureModel::<f64, Gaussian>::flat(vec![g1, g2]);

        assert_eq!(m.components.len(), 2);
        assert_eq!(m.weights.len(), 2);

        assert_relative_eq!(m.weights[0], 0.5, epsilon = TOL);
        assert_relative_eq!(m.weights[1], 0.5, epsilon = TOL);
    }

    #[test]
    fn new_categorical() {
        let c1: Categorical = Categorical::uniform(3);
        let c2: Categorical = Categorical::uniform(3);

        let m = MixtureModel::<u8, Categorical>::flat(vec![c1, c2]);

        assert_eq!(m.components.len(), 2);
        assert_eq!(m.weights.len(), 2);

        assert_relative_eq!(m.weights[0], 0.5, epsilon = TOL);
        assert_relative_eq!(m.weights[1], 0.5, epsilon = TOL);
    }

    #[test]
    fn categorical_argmax() {
        let ln_1 = (0.1 as f64).ln();
        let ln_2 = (0.2 as f64).ln();
        let ln_7 = (0.7 as f64).ln();
        let c1: Categorical =
            Categorical::from_ln_weights(vec![ln_1, ln_2, ln_7]).unwrap();
        let c2: Categorical =
            Categorical::from_ln_weights(vec![ln_2, ln_7, ln_1]).unwrap();

        let m = MixtureModel::<u8, Categorical>::flat(vec![c1, c2]);

        let x = m.mode().unwrap();

        assert_eq!(x, 1);
    }

    #[test]
    fn categorical_argmax_singleton() {
        let g = Gaussian::new(0.0, 1.0).unwrap();
        let m = MixtureModel::<f64, Gaussian>::flat(vec![g]);

        let x = m.mode().unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1E-10);
    }

    #[test]
    fn categorical_argmax_dual() {
        let g1 = Gaussian::new(0.0, 1.0).unwrap();
        let g2 = Gaussian::new(2.0, 3.0).unwrap();
        let m = MixtureModel::<f64, Gaussian>::flat(vec![g1, g2]);

        let x = m.mode().unwrap();

        assert_relative_eq!(x, 0.058422259659025054, epsilon = 1E-5);
    }
}

extern crate rand;

use std::marker::PhantomData;

use self::rand::Rng;

use misc::{pflip, logsumexp};
use dist::traits::{Argmax, Distribution, Entropy, RandomVariate};
use dist::{Categorical, Gaussian};
use optimize::{fmin_brute, fmin_bounded};


pub struct MixtureModel<M, T>
    where M: RandomVariate<T> + Distribution<T> + Entropy
{
    components: Vec<M>,
    weights: Vec<f64>,
    _phantom: PhantomData<T>,
}


impl<M, T> MixtureModel<M, T>
    where M: RandomVariate<T> + Distribution<T> + Entropy
{
    pub fn flat(components: Vec<M>) -> Self {
        let k = components.len();
        let weights = vec![1.0/(k as f64); k];

        MixtureModel{components: components,
                     weights: weights,
                     _phantom: PhantomData}
    }

    pub fn log_weights(&self) -> Vec<f64> {
        self.weights.iter().map(|w| w.ln()).collect()
    }

    pub fn sample(&self, n: usize, mut rng: &mut Rng) -> Vec<T> {
        let mut xs = Vec::with_capacity(n);
        for k in pflip(&self.weights, n, &mut rng) {
            xs.push(self.components[k].draw(&mut rng));
        }
        xs
    }

    pub fn loglike(&self, x: &T) -> f64 {
        let cpnt_loglikes: Vec<f64> = self.components
            .iter()
            .zip(&self.weights)
            .map(|(cpnt, w)| w.ln() + cpnt.loglike(x))
            .collect();

        logsumexp(&cpnt_loglikes)
    }

    pub fn loglikes(&self, xs: &[T]) -> Vec<f64> {
        xs.iter().map(|x| self.loglike(x)).collect()
    }

    pub fn entropy(&self, n_samples: usize, mut rng: &mut Rng) -> f64 {
        let xs = self.sample(n_samples, &mut rng);
        let logn = (n_samples as f64).ln();
        self.loglikes(&xs).iter().fold(0.0, |acc, ll| acc + ll) - logn
    }

    /// Normalized Jensen-Shannon divergence
    pub fn js_divergence(&self, n_samples: usize, mut rng: &mut Rng) -> f64 {
        let log_weights = self.log_weights();
        let h_all = self.entropy(n_samples, &mut rng);
        let h_sum = self.components
            .iter()
            .zip(log_weights)
            .fold(0.0, |acc, (cpnt, logw)| acc + logw + cpnt.entropy());

        let k = self.weights.len() as f64;

        (h_all - h_sum) / k.ln()
    }
}


impl Argmax for MixtureModel<Gaussian, f64> {
    type Output = f64;
    fn argmax(&self) -> f64 {
        let k = self.components.len();
        if k == 1 {
            self.components[0].mu
        } else {
            let _means: Vec<f64> = self.components.iter()
                .map(|cpnt| cpnt.mu)
                .collect();
            let (m0, means) = _means.split_first().unwrap();
            let a = means.iter().fold(m0, |min, x| if x < min {x} else {min});
            let b = means.iter().fold(m0, |max, x| if x > max {x} else {max});

            let n_grid = 50.max(k);
            let x0 = fmin_brute(|x| -self.loglike(&x), (*a, *b), n_grid);

            let eps = (b - a) / (n_grid as f64);
            let xatol = 2.0*eps * 1.0E-5;
            let bounds = (x0 - eps, x0 + eps);

            fmin_bounded(|x| -self.loglike(&x), bounds, Some(xatol), None,
                         Some(x0))
        }
    }
}


// FIXME: make generic to unisgned types
impl Argmax for MixtureModel<Categorical<u8>, u8> {
    type Output = u8;
    fn argmax(&self) -> u8 {
        let k = self.components[0].log_weights.len();
        let pairs: Vec<(u8, f64)> = (0..k).map(|x| {
            let xi = x as u8;
            (xi, self.loglike(&xi))
        }).collect();

        let (first, rest) = pairs.split_first().unwrap();

        rest.iter().fold(first, |current, nxt| {
            if nxt.1 > current.1 {nxt} else {current}
        }).0
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;
    
    #[test]
    fn new_gaussian() {
        let g1 = Gaussian::new(0.0, 1.0);
        let g2 = Gaussian::new(2.0, 3.0);

        let m = MixtureModel::flat(vec![g1, g2]);

        assert_eq!(m.components.len(), 2);
        assert_eq!(m.weights.len(), 2);

        assert_relative_eq!(m.weights[0], 0.5, epsilon=TOL);
        assert_relative_eq!(m.weights[1], 0.5, epsilon=TOL);
    }

    #[test]
    fn new_categorical() {
        let c1: Categorical<u8> = Categorical::flat(3);
        let c2: Categorical<u8> = Categorical::flat(3);

        let m = MixtureModel::flat(vec![c1, c2]);

        assert_eq!(m.components.len(), 2);
        assert_eq!(m.weights.len(), 2);

        assert_relative_eq!(m.weights[0], 0.5, epsilon=TOL);
        assert_relative_eq!(m.weights[1], 0.5, epsilon=TOL);
    }

    #[test]
    fn categorical_argmax() {
        let ln_1 = (0.1 as f64).ln();
        let ln_2 = (0.2 as f64).ln();
        let ln_7 = (0.7 as f64).ln();
        let c1: Categorical<u8> = Categorical::new(vec![ln_1, ln_2, ln_7]);
        let c2: Categorical<u8> = Categorical::new(vec![ln_2, ln_7, ln_1]);

        let m = MixtureModel::flat(vec![c1, c2]);

        let x = m.argmax();

        assert_eq!(x, 1);
    }

    #[test]
    fn categorical_argmax_singleton() {
        let g = Gaussian::new(0.0, 1.0);
        let m = MixtureModel::flat(vec![g]);

        let x = m.argmax();
        assert_relative_eq!(x, 0.0, epsilon=1E-10);
    }

    #[test]
    fn categorical_argmax_dual() {
        let g1 = Gaussian::new(0.0, 1.0);
        let g2 = Gaussian::new(2.0, 3.0);
        let m = MixtureModel::flat(vec![g1, g2]);

        let x = m.argmax();

        assert_relative_eq!(x, 0.058422259659025054, epsilon=1E-5);
    }
}

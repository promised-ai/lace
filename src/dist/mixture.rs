extern crate rand;

use std::marker::PhantomData;
use self::rand::Rng;
use misc::pflip;
use dist::traits::{Distribution, Entropy, RandomVariate};


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

    pub fn loglike(&self, xs: &[T]) -> Vec<f64> {
        let log_weights = self.log_weights();
        xs.iter().map(|x| {
            self.components
                .iter()
                .zip(&log_weights)
                .fold(0.0, |acc, (cpnt, logw)| acc + logw + cpnt.loglike(x))
        }).collect()
    }

    pub fn entropy(&self, n_samples: usize, mut rng: &mut Rng) -> f64 {
        let xs = self.sample(n_samples, &mut rng);
        let logn = (n_samples as f64).ln();
        self.loglike(&xs).iter().fold(0.0, |acc, ll| acc + ll) - logn
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

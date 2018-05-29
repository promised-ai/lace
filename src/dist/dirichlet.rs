extern crate rand;

use self::rand::distributions::{Gamma, IndependentSample};
use self::rand::Rng;
use dist::traits::Distribution;
use dist::traits::KlDivergence;
use dist::traits::Moments;
use dist::traits::RandomVariate;
use special::gamma::gammaln;

// Standard (uniform) Dirichlet
// ----------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymmetricDirichlet {
    pub alpha: f64,
    pub k: usize,
}

impl SymmetricDirichlet {
    pub fn new(alpha: f64, k: usize) -> Self {
        SymmetricDirichlet { alpha: alpha, k: k }
    }

    pub fn jeffereys(k: usize) -> Self {
        SymmetricDirichlet {
            alpha: 1.0 / k as f64,
            k: k,
        }
    }
}

impl RandomVariate<Vec<f64>> for SymmetricDirichlet {
    fn draw(&self, mut rng: &mut Rng) -> Vec<f64> {
        // TODO: offload to Gamma distribution
        let gamma = Gamma::new(self.alpha, 1.0);
        let xs: Vec<f64> =
            (0..self.k).map(|_| gamma.ind_sample(&mut rng)).collect();
        let z = xs.iter().fold(0.0, |acc, x| acc + x);
        xs.iter().map(|x| x / z).collect()
    }
}

impl Distribution<Vec<f64>> for SymmetricDirichlet {
    fn log_normalizer(&self) -> f64 {
        let kf: f64 = self.k as f64;
        gammaln(kf * self.alpha) - kf * gammaln(self.alpha)
    }

    fn unnormed_loglike(&self, x: &Vec<f64>) -> f64 {
        x.iter()
            .fold(0.0, |logf, &xi| logf + (self.alpha - 1.0) * xi.ln())
    }
}

impl Moments<Vec<f64>, Vec<f64>> for SymmetricDirichlet {
    fn mean(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha * (self.k as f64);
        vec![self.alpha / sum_alpha; self.k]
    }

    fn var(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha * (self.k as f64);
        let numer = self.alpha * (sum_alpha - self.alpha);
        let denom = sum_alpha * sum_alpha * (sum_alpha - 1.0);
        vec![numer / denom; self.k]
    }
}

impl KlDivergence for SymmetricDirichlet {
    fn kl_divergence(&self, _other: &Self) -> f64 {
        unimplemented!("Requires Digamma function");
    }
}

// Standard Dirichlet
// ------------------
pub struct Dirichlet {
    pub alpha: Vec<f64>,
}

impl Dirichlet {
    pub fn new(alpha: Vec<f64>) -> Self {
        Dirichlet { alpha: alpha }
    }

    pub fn jeffereys(k: usize) -> Self {
        Dirichlet {
            alpha: vec![1.0 / k as f64; k],
        }
    }
}

impl RandomVariate<Vec<f64>> for Dirichlet {
    fn draw(&self, mut rng: &mut Rng) -> Vec<f64> {
        // TODO: offload to Gamma distribution
        let xs: Vec<f64> = self
            .alpha
            .iter()
            .map(|a| {
                let gamma = Gamma::new(*a, 1.0);
                gamma.ind_sample(&mut rng)
            })
            .collect();

        let z = xs.iter().fold(0.0, |acc, x| acc + x);
        xs.iter().map(|x| x / z).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn symmetric_draw_weights_should_all_be_less_than_one() {
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let weights = symdir.draw(&mut rng);

        assert!(weights.iter().all(|lw| *lw < 1.0));
    }

    #[test]
    fn symmetric_draw_weights_should_sum_to_one() {
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let weights = symdir.draw(&mut rng);

        let sum_weights = weights.iter().sum();
        assert_relative_eq!(sum_weights, 1.0, epsilon = 10E-10);
    }

    #[test]
    fn symmetric_draw_weights_should_be_unique() {
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let weights = symdir.draw(&mut rng);

        assert_relative_ne!(weights[0], weights[1], epsilon = 10e-10);
        assert_relative_ne!(weights[1], weights[2], epsilon = 10e-10);
        assert_relative_ne!(weights[2], weights[3], epsilon = 10e-10);
        assert_relative_ne!(weights[0], weights[2], epsilon = 10e-10);
        assert_relative_ne!(weights[0], weights[3], epsilon = 10e-10);
        assert_relative_ne!(weights[1], weights[3], epsilon = 10e-10);
    }
}

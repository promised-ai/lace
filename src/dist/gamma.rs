extern crate rand;
extern crate serde;

use std::f64;

use self::rand::Rng;

use dist::traits::Distribution;
use dist::traits::Moments;
use dist::traits::RandomVariate;
use special::gammaln;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> Gamma {
        Gamma {
            shape: shape,
            rate: rate,
        }
    }
}

impl RandomVariate<f64> for Gamma {
    fn draw(&self, rng: &mut impl Rng) -> f64 {
        // The rand Gamma is parameterized by shape instead of rate
        let g = rand::distributions::Gamma::new(self.shape, 1.0 / self.rate);
        rng.sample(g)
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<f64> {
        let g = rand::distributions::Gamma::new(self.shape, 1.0 / self.rate);
        (0..n).map(|_| rng.sample(g)).collect()
    }
}

impl Distribution<f64> for Gamma {
    fn log_normalizer(&self) -> f64 {
        0.0
    }

    fn unnormed_loglike(&self, x: &f64) -> f64 {
        let a = self.shape;
        let b = self.rate;
        a * b.ln() - gammaln(a) + (a - 1.0) * x.ln() - (b * x)
    }
}

impl Moments<f64, f64> for Gamma {
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    fn var(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }
}

#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn gamma_new() {
        let g = Gamma::new(1.2, 3.4);
        assert_relative_eq!(g.shape, 1.2, epsilon = TOL);
        assert_relative_eq!(g.rate, 3.4, epsilon = TOL);
    }

    #[test]
    fn gamma_mean() {
        let g = Gamma::new(1.2, 3.4);
        assert_relative_eq!(g.mean(), 1.2 / 3.4, epsilon = TOL);
    }

    #[test]
    fn gamma_var() {
        let g = Gamma::new(1.2, 3.4);
        assert_relative_eq!(g.var(), 1.2 / (3.4 * 3.4), epsilon = TOL);
    }

    #[test]
    fn gamma_loglike_1() {
        let g = Gamma::new(1.0, 1.0);
        assert_relative_eq!(g.loglike(&1.5), -1.5, epsilon = TOL);
    }

    #[test]
    fn gamma_loglike_2() {
        let g = Gamma::new(1.2, 3.4);
        assert_relative_eq!(g.loglike(&1.5), -3.465002370428512, epsilon = TOL);
    }
}

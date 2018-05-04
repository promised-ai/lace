extern crate rand;
extern crate serde;

use std::f64;

use self::rand::Rng;
use self::rand::distributions::IndependentSample;

use dist::traits::Distribution;
use dist::traits::RandomVariate;
use special::gammaln;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InvGamma {
    pub shape: f64,
    pub rate: f64,
}

impl InvGamma {
    pub fn new(shape: f64, rate: f64) -> Self {
        InvGamma {
            shape: shape,
            rate: rate,
        }
    }
}

impl RandomVariate<f64> for InvGamma {
    fn draw(&self, mut rng: &mut Rng) -> f64 {
        // The rand Gamma is parameterized by shape instead of rate
        let g = rand::distributions::Gamma::new(self.shape, 1.0 / self.rate);
        1.0 / g.ind_sample(&mut rng)
    }

    fn sample(&self, n: usize, mut rng: &mut Rng) -> Vec<f64> {
        let g = rand::distributions::Gamma::new(self.shape, self.rate);
        (0..n)
            .map(|_| 1.0 / g.ind_sample(&mut rng))
            .collect()
    }
}

impl Distribution<f64> for InvGamma {
    fn log_normalizer(&self) -> f64 {
        0.0
    }

    fn unnormed_loglike(&self, x: &f64) -> f64 {
        let a = self.shape;
        let b = self.rate;
        a * b.ln() - gammaln(a) - (a + 1.0) * x.ln() - (b / x)
    }
}

// TODO: These are not always defined
// impl Moments<f64, f64> for Gamma {
//     fn mean(&self) -> f64 {
//         self.rate / (self.shape - 1.0)
//     }

//     fn var(&self) -> f64 {
//         let c = self.shape - 1.0;
//         self.rate / (c*c * (self.shape - 2.0))
//     }
// }

#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn invgamma_new() {
        let ig = InvGamma::new(1.2, 3.4);
        assert_relative_eq!(ig.shape, 1.2, epsilon = TOL);
        assert_relative_eq!(ig.rate, 3.4, epsilon = TOL);
    }

    #[test]
    fn invgamma_loglike_1() {
        let ig = InvGamma::new(1.0, 1.0);
        assert_relative_eq!(
            ig.loglike(&1.5),
            -1.4775968828829953,
            epsilon = TOL
        );
    }

    #[test]
    fn invgamma_loglike_2() {
        let ig = InvGamma::new(1.2, 3.4);
        assert_relative_eq!(
            ig.loglike(&1.5),
            -1.6047852965547733,
            epsilon = TOL
        );
    }
}

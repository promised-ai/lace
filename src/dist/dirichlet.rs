extern crate rand;

use self::rand::Rng;
use self::rand::distributions::{Gamma, IndependentSample};
use dist::traits::Distribution;
use dist::traits::RandomVariate;
use dist::traits::Moments;
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
        SymmetricDirichlet{alpha: alpha, k: k}
    }

    pub fn jeffereys(k: usize) -> Self{
        SymmetricDirichlet{alpha: 1.0 / k as f64, k: k}
    }
}


impl RandomVariate<Vec<f64>> for SymmetricDirichlet {
    fn draw(&self, mut rng: &mut Rng) -> Vec<f64> {
        // TODO: offload to Gamma distribution
        let gamma = Gamma::new(self.alpha, 1.0);
        let xs: Vec<f64> = (0..self.k).map(|_| {
            gamma.ind_sample(&mut rng)
        }).collect();
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
        x.iter().fold(0.0, |logf, &xi| logf + (self.alpha - 1.0) * xi.ln())
    }
}


impl Moments<Vec<f64>, Vec<f64>> for SymmetricDirichlet {
    fn mean(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha * (self.k as f64);
        vec![self.alpha/sum_alpha; self.k]
    }

    fn var(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha * (self.k as f64);
        let numer = self.alpha * (sum_alpha - self.alpha);
        let denom = sum_alpha * sum_alpha * (sum_alpha - 1.0);
        vec![numer/denom; self.k]
    }
}


// Standard Dirichlet
// ------------------
pub struct Dirichlet {
    pub alpha: Vec<f64>,
}

impl Dirichlet {
    pub fn new(alpha: Vec<f64>) -> Self {
        Dirichlet{alpha: alpha}
    }

    pub fn jeffereys(k: usize) -> Self {
        Dirichlet{alpha: vec![1.0 / k as f64; k]}
    }
}


impl RandomVariate<Vec<f64>> for Dirichlet {
    fn draw(&self, mut rng: &mut Rng) -> Vec<f64> {
        // TODO: offload to Gamma distribution
        let xs: Vec<f64> = self.alpha.iter().map(|a| {
            let gamma = Gamma::new(*a, 1.0);
            gamma.ind_sample(&mut rng)
        }).collect();

        let z = xs.iter().fold(0.0, |acc, x| acc + x);
        xs.iter().map(|x| x / z).collect()
    }
}

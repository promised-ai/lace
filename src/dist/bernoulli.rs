extern crate rand;

use self::rand::Rng;
use dist::traits::Cdf;
use dist::traits::Distribution;
use dist::traits::RandomVariate;
use dist::traits::Entropy;
use dist::traits::Moments;


pub struct Bernoulli {
    pub p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Bernoulli {
        Bernoulli{p: p}
    }

    fn q(&self) -> f64 {
        1.0 - self.p
    }
}


impl RandomVariate<bool> for Bernoulli {
    fn draw(&self, rng: &mut Rng) -> bool {
        rng.next_f64() < self.p
    }
}


impl Distribution<bool> for Bernoulli {
    fn unnormed_loglike(&self, x: &bool) -> f64 {
        if *x {
            self.p.ln()
        } else {
            self.q().ln()
        }
    }

    fn log_normalizer(&self) -> f64 { 0.0 }
}


impl Moments<f64, f64> for Bernoulli {
    fn mean(&self) -> f64 {
        self.p
    }
    fn var(&self) -> f64 {
        self.p * self.q()
    }
}


impl Cdf<bool> for Bernoulli {
    fn cdf(&self, x: &bool) -> f64 {
        if *x {
            1.0
        } else {
            self.q()
        }
    }
}

impl Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        -(self.p * self.p.ln() + self.q() * self.q().ln())
    }
}

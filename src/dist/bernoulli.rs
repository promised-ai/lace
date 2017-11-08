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


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean() {
        let bern = Bernoulli::new(0.6);
        assert_approx_eq!(bern.mean(), 0.6, 10E-8);
    }


    #[test]
    fn variance() {
        let bern = Bernoulli::new(0.3);
        assert_approx_eq!(bern.var(), 0.3 * 0.7, 10E-8);
    }


    #[test]
    fn like_true_should_be_log_of_p() {
        let bern1 = Bernoulli::new(0.5);
        assert_approx_eq!(bern1.like(&true), 0.5, 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_approx_eq!(bern2.like(&true), 0.95, 10E-8);
    }


    #[test]
    fn like_false_should_be_log_of_q() {
        let bern1 = Bernoulli::new(0.5);
        assert_approx_eq!(bern1.like(&false), 0.5, 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_approx_eq!(bern2.like(&false), 0.05, 10E-8);
    }


    #[test]
    fn loglike_true_should_be_log_of_p() {
        let bern1 = Bernoulli::new(0.5);
        assert_approx_eq!(bern1.loglike(&true), (0.5 as f64).ln(), 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_approx_eq!(bern2.loglike(&true), (0.95 as f64).ln(), 10E-8);
    }


    #[test]
    fn loglike_false_should_be_log_of_q() {
        let bern1 = Bernoulli::new(0.5);
        assert_approx_eq!(bern1.loglike(&false), (0.5 as f64).ln(), 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_approx_eq!(bern2.loglike(&false), (0.05 as f64).ln(), 10E-8);
    }
}

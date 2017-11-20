extern crate rand;

use self::rand::Rng;
use dist::traits::Cdf;
use dist::traits::Distribution;
use dist::traits::AccumScore;
use dist::traits::SufficientStatistic;
use dist::traits::HasSufficientStatistic;
use dist::traits::RandomVariate;
use dist::traits::Entropy;
use dist::traits::Moments;


pub struct Bernoulli {
    pub p: f64,
    pub suffstats: BernoulliSuffStats,
}


pub struct BernoulliSuffStats {
    pub n: u64,
    pub k: u64,
}

impl BernoulliSuffStats {
    pub fn new() -> Self {
        BernoulliSuffStats{n: 0, k: 0}
    }
}

impl Bernoulli {
    pub fn new(p: f64) -> Bernoulli {
        Bernoulli{p: p, suffstats: BernoulliSuffStats::new()}
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


impl AccumScore<bool> for Bernoulli {}


impl SufficientStatistic<bool> for BernoulliSuffStats {
    fn observe(&mut self, x: &bool) {
        self.n += 1;
        self.k += *x as u64;
    }

    fn unobserve(&mut self, x: &bool) {
        if self.n == 0 {
            panic!["No observations to unobserve."]
        }
        self.n -= 1;
        self.k -= *x as u64;
   }
}


// TODO: make this a macro
impl HasSufficientStatistic<bool> for Bernoulli {
    fn observe(&mut self, x: &bool) {
        self.suffstats.observe(x);
    }

    fn unobserve(&mut self, x: &bool) {
        self.suffstats.unobserve(x);
    }
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
        assert_relative_eq!(bern.mean(), 0.6, epsilon = 10E-8);
    }


    #[test]
    fn variance() {
        let bern = Bernoulli::new(0.3);
        assert_relative_eq!(bern.var(), 0.3 * 0.7, epsilon = 10E-8);
    }


    #[test]
    fn like_true_should_be_log_of_p() {
        let bern1 = Bernoulli::new(0.5);
        assert_relative_eq!(bern1.like(&true), 0.5, epsilon = 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_relative_eq!(bern2.like(&true), 0.95, epsilon = 10E-8);
    }


    #[test]
    fn like_false_should_be_log_of_q() {
        let bern1 = Bernoulli::new(0.5);
        assert_relative_eq!(bern1.like(&false), 0.5, epsilon = 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_relative_eq!(bern2.like(&false), 0.05, epsilon = 10E-8);
    }


    #[test]
    fn loglike_true_should_be_log_of_p() {
        let bern1 = Bernoulli::new(0.5);
        assert_relative_eq!(bern1.loglike(&true), (0.5 as f64).ln(),
                            epsilon = 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_relative_eq!(bern2.loglike(&true), (0.95 as f64).ln(),
                            epsilon = 10E-8);
    }


    #[test]
    fn loglike_false_should_be_log_of_q() {
        let bern1 = Bernoulli::new(0.5);
        assert_relative_eq!(bern1.loglike(&false), (0.5 as f64).ln(),
                            epsilon = 10E-8);

        let bern2 = Bernoulli::new(0.95);
        assert_relative_eq!(bern2.loglike(&false), (0.05 as f64).ln(),
                     epsilon = 10E-8);
    }
}

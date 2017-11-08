extern crate rand;

use std::f64;

use self::rand::Rng;
use self::rand::distributions::Normal;
use self::rand::distributions::IndependentSample;
use dist::traits::Cdf;
use dist::traits::Distribution;
use dist::traits::RandomVariate;
use dist::traits::Entropy;
use dist::traits::InverseCdf;
use dist::traits::Moments;
use dist::traits::Mode;

use special::{erf, erfinv};


const HALF_LOG_2PI: f64 = 0.918938533204672669540968854562379419803619384766;
const HALF_LOG_2PI_E: f64 = 1.418938533204672669540968854562379419803619384766;
const SQRT_PI: f64 = 1.772453850905515881919427556567825376987457275391;


pub struct Gaussian {
    pub mu: f64,
    pub sigma: f64,
}


impl Gaussian {
    pub fn new(mu: f64, sigma: f64) -> Gaussian {
        Gaussian {mu: mu, sigma: sigma}
    }

    pub fn standard() -> Gaussian {
        Gaussian {mu: 0.0, sigma: 1.0}
    }
}


impl RandomVariate<f64> for Gaussian {
    fn draw(&self, mut rng: &mut Rng) -> f64 {
        let g = Normal::new(self.mu, self.sigma);
        g.ind_sample(&mut rng)
    }

    fn sample(&self, n: usize, mut rng: &mut Rng) -> Vec<f64> {
        let g = Normal::new(self.mu, self.sigma);
        (0..n).map(|_| g.ind_sample(&mut rng)).collect()
    }
}


impl Distribution<f64> for Gaussian {

    fn log_normalizer(&self) -> f64 {
        HALF_LOG_2PI
    }

    fn unnormed_loglike(&self, x: &f64) -> f64 {
        let term = (x - self.mu)/self.sigma;
         -self.sigma.ln() - 0.5 * term * term
    }
}



impl Cdf<f64> for Gaussian {
    fn cdf(&self, x: &f64) -> f64 {
        0.5*(1.0 + erf((x - self.mu)/(self.sigma * SQRT_PI)))
    }
}


impl InverseCdf<f64> for Gaussian {
    fn invcdf(&self, p: &f64) -> f64 {
        if (*p <= 0.0) || (1.0 <= *p) {
            panic!("P out of range");
        }
        self.mu + self.sigma * f64::consts::SQRT_2 * erfinv(2.0**p-1.0)
    }
}


impl Moments<f64, f64> for Gaussian {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn var(&self) -> f64 {
        self.sigma * self.sigma
    }
}


impl Mode<f64> for Gaussian {
    fn mode(&self) -> f64 {
        self.mu
    }
}


impl Entropy for Gaussian {
    fn entropy(&self) -> f64 {
        HALF_LOG_2PI_E + self.sigma.ln()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8; 

    #[test]
    fn gaussian_new() {
        let gauss = Gaussian::new(1.2, 3.0);

        assert_approx_eq!(gauss.mu, 1.2, TOL);
        assert_approx_eq!(gauss.sigma, 3.0, TOL);
    }


    #[test]
    fn gaussian_standard() {
        let gauss = Gaussian::standard();

        assert_approx_eq!(gauss.mu, 0.0, TOL);
        assert_approx_eq!(gauss.sigma, 1.0, TOL);
    }


    #[test]
    fn gaussian_moments() {
        let gauss1 = Gaussian::standard();

        assert_approx_eq!(gauss1.mean(), 0.0, TOL);
        assert_approx_eq!(gauss1.var(), 1.0, TOL);

        let gauss2 = Gaussian::new(3.4, 0.5);

        assert_approx_eq!(gauss2.mean(), 3.4, TOL);
        assert_approx_eq!(gauss2.var(), 0.25, TOL);
    }


    #[test]
    fn gaussian_sample_length() {
        let mut rng = rand::thread_rng();
        let gauss = Gaussian::standard();
        let xs: Vec<f64> = gauss.sample(10, &mut rng);
        assert_eq!(xs.len(), 10);
    }


    #[test]
    fn gaussian_standard_loglike() {
        let gauss = Gaussian::standard();
        assert_approx_eq!(gauss.loglike(&0.0), -0.91893853320467267, TOL);
        assert_approx_eq!(gauss.loglike(&2.1), -3.1239385332046727, TOL);
    }


    #[test]
    fn gaussian_nonstandard_loglike() {
        let gauss = Gaussian::new(-1.2, 0.33);

        assert_approx_eq!(gauss.loglike(&-1.2), 0.18972409131693846, TOL);
        assert_approx_eq!(gauss.loglike(&0.0), -6.4218461566169447, TOL);
    }
}

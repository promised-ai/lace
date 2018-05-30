extern crate rand;
extern crate serde;

use rayon::prelude::*;
use std::f64;

use self::rand::distributions::Normal;
use self::rand::Rng;
use dist::traits::AccumScore;
use dist::traits::Argmax;
use dist::traits::Cdf;
use dist::traits::Distribution;
use dist::traits::Entropy;
use dist::traits::HasSufficientStatistic;
use dist::traits::InverseCdf;
use dist::traits::KlDivergence;
use dist::traits::Mode;
use dist::traits::Moments;
use dist::traits::RandomVariate;
use dist::traits::SufficientStatistic;

use special::{erf, erfinv};

const HALF_LOG_2PI: f64 = 0.918938533204672669540968854562379419803619384766;
const HALF_LOG_2PI_E: f64 = 1.418938533204672669540968854562379419803619384766;
const SQRT_PI: f64 = 1.772453850905515881919427556567825376987457275391;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gaussian {
    pub mu: f64,
    pub sigma: f64,
    #[serde(skip)]
    pub suffstats: GaussianSuffStats,
}

impl Gaussian {
    pub fn new(mu: f64, sigma: f64) -> Gaussian {
        Gaussian {
            mu: mu,
            sigma: sigma,
            suffstats: GaussianSuffStats::new(),
        }
    }

    pub fn standard() -> Gaussian {
        Gaussian {
            mu: 0.0,
            sigma: 1.0,
            suffstats: GaussianSuffStats::new(),
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct GaussianSuffStats {
    pub n: u64,
    pub sum_x: f64,
    pub sum_x_sq: f64,
}

impl GaussianSuffStats {
    pub fn new() -> Self {
        GaussianSuffStats::default()
    }
}

// TODO: use more numerically stable version
impl SufficientStatistic<f64> for GaussianSuffStats {
    fn observe(&mut self, x: &f64) {
        self.n += 1;
        self.sum_x += x;
        self.sum_x_sq += x * x;
    }

    fn unobserve(&mut self, x: &f64) {
        self.n -= 1;
        if self.n == 0 {
            self.sum_x = 0.0;
            self.sum_x_sq = 0.0;
        } else if self.n > 0 {
            self.sum_x -= x;
            self.sum_x_sq -= x * x;
        } else {
            panic!["No observations to unobserve."]
        }
    }
}

// TODO: make this a macro
impl HasSufficientStatistic<f64> for Gaussian {
    fn observe(&mut self, x: &f64) {
        self.suffstats.observe(x);
    }

    fn unobserve(&mut self, x: &f64) {
        self.suffstats.unobserve(x);
    }
}

impl RandomVariate<f64> for Gaussian {
    fn draw(&self, rng: &mut impl Rng) -> f64 {
        let g = Normal::new(self.mu, self.sigma);
        rng.sample(g)
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<f64> {
        let g = Normal::new(self.mu, self.sigma);
        (0..n).map(|_| rng.sample(g)).collect()
    }
}

impl Distribution<f64> for Gaussian {
    fn log_normalizer(&self) -> f64 {
        HALF_LOG_2PI
    }

    fn unnormed_loglike(&self, x: &f64) -> f64 {
        let term = (x - self.mu) / self.sigma;
        -self.sigma.ln() - 0.5 * term * term
    }
}

impl AccumScore<f64> for Gaussian {
    fn accum_score_par(
        &self,
        scores: &mut [f64],
        xs: &[f64],
        present: &[bool],
    ) {
        let mu = self.mu;
        let sigma = self.sigma;
        let log_z = -self.sigma.ln() - HALF_LOG_2PI;

        let xs_iter = xs.par_iter().zip_eq(present.par_iter());
        scores
            .par_iter_mut()
            .zip_eq(xs_iter)
            .for_each(|(score, (x, &r))| {
                if r {
                    // TODO: unnormed_loglike ?
                    let term = (x - mu) / sigma;
                    let loglike = -0.5 * term * term + log_z;
                    *score += loglike;
                }
            });
    }
}

impl Cdf<f64> for Gaussian {
    fn cdf(&self, x: &f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mu) / (self.sigma * SQRT_PI)))
    }
}

impl InverseCdf<f64> for Gaussian {
    fn invcdf(&self, p: &f64) -> f64 {
        if (*p <= 0.0) || (1.0 <= *p) {
            panic!("P out of range");
        }
        self.mu + self.sigma * f64::consts::SQRT_2 * erfinv(2.0 * *p - 1.0)
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

impl KlDivergence for Gaussian {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let m1 = self.mu;
        let m2 = other.mu;

        let s1 = self.sigma;
        let s2 = other.sigma;

        let term1 = s2.ln() - s1.ln();
        let term2 = (s1 * s1 + (m1 - m2) * (m1 - m2)) / (2.0 * s2 * s2);

        term1 + term2 - 0.5
    }
}

impl Argmax for Gaussian {
    type Output = f64;
    fn argmax(&self) -> f64 {
        self.mu
    }
}

#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn gaussian_new() {
        let gauss = Gaussian::new(1.2, 3.0);

        assert_relative_eq!(gauss.mu, 1.2, epsilon = TOL);
        assert_relative_eq!(gauss.sigma, 3.0, epsilon = TOL);
        assert_eq!(gauss.suffstats.n, 0);
        assert_relative_eq!(gauss.suffstats.sum_x, 0.0, epsilon = 10E-10);
        assert_relative_eq!(gauss.suffstats.sum_x_sq, 0.0, epsilon = 10E-10);
    }

    #[test]
    fn gaussian_standard() {
        let gauss = Gaussian::standard();

        assert_relative_eq!(gauss.mu, 0.0, epsilon = TOL);
        assert_relative_eq!(gauss.sigma, 1.0, epsilon = TOL);
    }

    #[test]
    fn gaussian_moments() {
        let gauss1 = Gaussian::standard();

        assert_relative_eq!(gauss1.mean(), 0.0, epsilon = TOL);
        assert_relative_eq!(gauss1.var(), 1.0, epsilon = TOL);

        let gauss2 = Gaussian::new(3.4, 0.5);

        assert_relative_eq!(gauss2.mean(), 3.4, epsilon = TOL);
        assert_relative_eq!(gauss2.var(), 0.25, epsilon = TOL);
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
        assert_relative_eq!(
            gauss.loglike(&0.0),
            -0.91893853320467267,
            epsilon = TOL
        );
        assert_relative_eq!(
            gauss.loglike(&2.1),
            -3.1239385332046727,
            epsilon = TOL
        );
    }

    #[test]
    fn gaussian_nonstandard_loglike() {
        let gauss = Gaussian::new(-1.2, 0.33);

        assert_relative_eq!(
            gauss.loglike(&-1.2),
            0.18972409131693846,
            epsilon = TOL
        );
        assert_relative_eq!(
            gauss.loglike(&0.0),
            -6.4218461566169447,
            epsilon = TOL
        );
    }

    #[test]
    fn gausssian_suffstat_observe_1() {
        let mut gauss = Gaussian::standard();
        gauss.observe(&2.0);

        assert_eq!(gauss.suffstats.n, 1);
        assert_relative_eq!(gauss.suffstats.sum_x, 2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            gauss.suffstats.sum_x_sq,
            4.0,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn gausssian_suffstat_observe_2() {
        let mut gauss = Gaussian::standard();
        gauss.observe(&2.0);
        gauss.observe(&4.0);

        assert_eq!(gauss.suffstats.n, 2);
        assert_relative_eq!(
            gauss.suffstats.sum_x,
            2.0 + 4.0,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            gauss.suffstats.sum_x_sq,
            4.0 + 16.0,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn gausssian_suffstat_unobserve_1() {
        let mut gauss = Gaussian::standard();
        gauss.observe(&2.0);
        gauss.observe(&4.0);
        gauss.unobserve(&4.0);

        assert_eq!(gauss.suffstats.n, 1);
        assert_relative_eq!(gauss.suffstats.sum_x, 2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            gauss.suffstats.sum_x_sq,
            4.0,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn gausssian_suffstat_unobserve_to_zero_resets_stats() {
        let mut gauss = Gaussian::standard();
        gauss.observe(&2.0);
        gauss.observe(&4.0);
        gauss.unobserve(&2.0);
        gauss.unobserve(&4.0);

        assert_eq!(gauss.suffstats.n, 0);
        assert_relative_eq!(gauss.suffstats.sum_x, 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            gauss.suffstats.sum_x_sq,
            0.0,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    #[should_panic]
    fn gaussian_suffstat_unobserv_empty_should_panic() {
        let mut gauss = Gaussian::standard();
        gauss.unobserve(&2.0);
    }

    #[test]
    fn serialize() {
        let gauss = Gaussian::new(2.1, 3.3);
        let yaml = serde_yaml::to_string(&gauss).unwrap();
        assert_eq!(yaml, "---\nmu: 2.1\nsigma: 3.3");
    }

    #[test]
    fn deserialize() {
        let yaml = "---\nmu: 3.1\nsigma: 2.2";

        let gauss: Gaussian = serde_yaml::from_str(&yaml).unwrap();
        assert_relative_eq!(gauss.mu, 3.1, epsilon = 10e-10);
        assert_relative_eq!(gauss.sigma, 2.2, epsilon = 10e-10);
        assert_eq!(gauss.suffstats.n, 0);
        assert_relative_eq!(gauss.suffstats.sum_x, 0.0, epsilon = 10e-10);
        assert_relative_eq!(gauss.suffstats.sum_x_sq, 0.0, epsilon = 10e-10);
    }

    #[test]
    fn argmax_should_be_mean() {
        let gauss = Gaussian::new(0.1, 1.2);
        assert_relative_eq!(gauss.argmax(), 0.1, epsilon = 10e-10);
    }
}

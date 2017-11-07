extern crate rand;

use self::rand::Rng;
use self::rand::distributions::{Gamma, Normal, IndependentSample};

use dist::traits::Distribution;
use dist::suffstats::{GaussianSuffStats, SufficientStatistic};

use dist::Gaussian;
use dist::Bernoulli;
use special::gammaln;


const LOG2: f64 = 0.69314718055994528622676398299518041312694549560546875;
const HALF_LOG_PI: f64 = 0.57236494292470008193873809432261623442173004150390625;
const HALF_LOG_2PI: f64 = 0.918938533204672669540968854562379419803619384766;


// TODO: rename file to priors
pub trait Prior<T, M>
    where M: Distribution<T>,
{
    fn posterior_draw(&self, data: &Vec<&T>, rng: &mut Rng) -> M;
    fn prior_draw(&self, rng: &mut Rng) -> M;
    fn marginal_score(&self, data: &Vec<&T>) -> f64;
    fn update_params(&mut self, components: &Vec<M>);

    fn draw(&self, data_opt: Option<&Vec<&T>>, mut rng: &mut Rng) -> M {
        match data_opt {
            Some(data) => self.posterior_draw(data, &mut rng),
            None       => self.prior_draw(&mut rng)
        }
    }

    // Not needed until split-merge or Gibbs implemented:
    // fn predictive_score(&self, x: &T, y: &Vec<T>) -> f64;
    // fn singleton_score(&self, y: &t) -> f64;
}


// Normmal, Inverse-Gamma prior for Normal data
// --------------------------------------------
pub struct NormalInverseGamma {
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
}


impl NormalInverseGamma {
    pub fn new(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalInverseGamma{m: m, r: r, s: s, v: v}
    }

    fn posterior_params(&self, suffstats: &GaussianSuffStats) -> Self {
        let r = self.r + (suffstats.n as f64);
        let v = self.v + (suffstats.n as f64);
        let m = (self.m * self.r + suffstats.sum_x) / r;
        let s = self.s + suffstats.sum_x_sq + self.r*self.m*self.m - r*m*m;
        NormalInverseGamma{m: m, r: r, s: s, v: v}
    }

    fn log_normalizer(r: f64, s: f64, v: f64) -> f64 {
        (v + 1.0)/2.0 * LOG2 + HALF_LOG_PI - 0.5 * r.ln() + gammaln(v/2.0)
    }

}


impl Prior<f64, Gaussian> for NormalInverseGamma {
    fn posterior_draw(&self, data: &Vec<&f64>, mut rng: &mut Rng) -> Gaussian {
        let mut suffstats = GaussianSuffStats::new();
        for x in data {
            suffstats.observe(x);
        }
        self.posterior_params(&suffstats).prior_draw(&mut rng)
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Gaussian {
        let rho = Gamma::new(self.v/2.0, self.s/2.0).ind_sample(&mut rng);
        let post_sigma = (1.0/(self.r*rho)).sqrt();
        let mu = Normal::new(self.m, post_sigma).ind_sample(&mut rng);

        Gaussian::new(mu, 1.0/rho.sqrt())
    }

    fn marginal_score(&self, data: &Vec<&f64>) -> f64 {
        let mut suffstats = GaussianSuffStats::new();
        for x in data {
            suffstats.observe(x);
        }
        let pr = self.posterior_params(&suffstats);
        let z0 = Self::log_normalizer(self.r, self.s, self.v);
        let zn = Self::log_normalizer(pr.r, pr.s, pr.v);
        -(suffstats.n as f64) * HALF_LOG_2PI + zn - z0
    }

    fn update_params(&mut self, components: &Vec<Gaussian>) {
        unimplemented!();
    }
}


// Beta prior for bernoulli likelihood
// -----------------------------------
struct BetaBernoulli {
    pub a: f64,
    pub b: f64,
}


impl Prior<bool, Bernoulli> for NormalInverseGamma {
    fn posterior_draw(&self, data: &Vec<&bool>, mut rng: &mut Rng) -> Bernoulli {
        unimplemented!();
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Bernoulli {
        unimplemented!();
    }

    fn marginal_score(&self, y: &Vec<&bool>) -> f64 {
        unimplemented!();
    }

    fn update_params(&mut self, components: &Vec<Bernoulli>) {
        unimplemented!();
    }
}

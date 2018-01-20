extern crate rand;

use std::f64::consts::LN_2;
use self::rand::Rng;
use self::rand::distributions::{Normal, IndependentSample};

use dist::prior::Prior;
use dist::{Gaussian, Gamma};
use dist::traits::{SufficientStatistic, RandomVariate, Distribution};
use dist::gaussian::GaussianSuffStats;
use special::gammaln;
use misc::{mean, var};
use misc::mh::mh_prior;


const HALF_LOG_PI: f64 = 0.57236494292470008193873809432261623442173004150390625;
const HALF_LOG_2PI: f64 = 0.918938533204672669540968854562379419803619384766;


// Normmal, Inverse-Gamma prior for Normal data
// --------------------------------------------
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NormalInverseGamma {
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
}

// Reference:
// https://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
impl NormalInverseGamma {
    pub fn new(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalInverseGamma{m: m, r: r, s: s, v: v}
    }

    // TODO: implement for f32 and f64 data
    pub fn from_data(xs: &[f64]) -> Self {
        NormalInverseGamma{m: mean(xs), r: 1.0, s: var(xs), v: 1.0}
    }

    fn posterior_params(&self, suffstats: &GaussianSuffStats) -> Self {
        let r = self.r + (suffstats.n as f64);
        let v = self.v + (suffstats.n as f64);
        let m = (self.m * self.r + suffstats.sum_x) / r;
        let s = self.s + suffstats.sum_x_sq + self.r*self.m*self.m - r*m*m;
        assert!(s > 0.0);
        NormalInverseGamma{m: m, r: r, s: s, v: v}
    }

    fn log_normalizer(r: f64, s: f64, v: f64) -> f64 {
        (v + 1.0)/2.0 * LN_2
            + HALF_LOG_PI
            - 0.5 * r.ln()
            - (v/2.0) * s.ln()
            + gammaln(v/2.0)
    }
}


impl Prior<f64, Gaussian> for NormalInverseGamma {
    fn loglike(&self, model: &Gaussian)-> f64 {
        let rho = 1.0/(model.sigma * model.sigma);
        let logp_rho = Gamma::new(self.v/2.0, self.s/2.0).loglike(&rho);
        let prior_sigma = (1.0/(self.r*rho)).sqrt();
        let logp_mu = Gaussian::new(self.m, prior_sigma).loglike(&model.mu);
        logp_rho + logp_mu
    }

    fn posterior_draw(&self, data: &[f64], mut rng: &mut Rng) -> Gaussian {
        assert!(!data.is_empty());
        let mut suffstats = GaussianSuffStats::new();
        for x in data {
            suffstats.observe(x);
        }
        assert_eq!(suffstats.n, data.len() as u64);
        self.posterior_params(&suffstats).prior_draw(&mut rng)
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Gaussian {
        let rho = Gamma::new(self.v/2.0, self.s/2.0).draw(&mut rng);
        let post_sigma = (1.0/(self.r*rho)).sqrt();
        let mu = Normal::new(self.m, post_sigma).ind_sample(&mut rng);

        Gaussian::new(mu, 1.0/rho.sqrt())
    }

    fn marginal_score(&self, data: &[f64]) -> f64 {
        let mut suffstats = GaussianSuffStats::new();
        for x in data {
            suffstats.observe(x);
        }
        let pr = self.posterior_params(&suffstats);
        let z0 = Self::log_normalizer(self.r, self.s, self.v);
        let zn = Self::log_normalizer(pr.r, pr.s, pr.v);
        -(suffstats.n as f64) * HALF_LOG_2PI + zn - z0
    }

    fn update_params(&mut self, components: &[Gaussian], mut rng: &mut Rng) {
        // update m
        let new_m: f64;
        let new_r: f64;
        let new_s: f64;
        let new_v: f64;
        {
            let draw = |mut rng: &mut Rng| {
                Normal::new(0.0, 1.0).ind_sample(&mut rng)
            };
            let f = |m: &f64| {
                let nig = NormalInverseGamma::new(*m, self.r, self.s, self.v);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_m = mh_prior(f, draw, 50, &mut rng);
        }
        self.m = new_m;
        
        // update r
        {
            let draw = |mut rng: &mut Rng| {
                1.0 / Gamma::new(1.0, 1.0).draw(&mut rng)
            };
            let f = |r: &f64| {
                let nig = NormalInverseGamma::new(self.m, *r, self.s, self.v);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_r = mh_prior(f, draw, 50, &mut rng);
        }
        self.r = new_r;

        // update s
        {
            let draw = |mut rng: &mut Rng| {
                1.0 / Gamma::new(1.0, 1.0).draw(&mut rng)
            };
            let f = |s: &f64| {
                let nig = NormalInverseGamma::new(self.m, self.r, *s, self.v);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_s = mh_prior(f, draw, 50, &mut rng);
        }
        self.s = new_s;

        // update v
        {
            let draw = |mut rng: &mut Rng| {
                1.0 / Gamma::new(1.0, 1.0).draw(&mut rng)
            };
            let f = |v: &f64| {
                let nig = NormalInverseGamma::new(self.m, self.r, self.s, *v);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_v = mh_prior(f, draw, 50, &mut rng);
        }
        self.v = new_v;
    }
}

// Hyperprior for later?
// --------------------
// TODO: I really don't like doing it like this. It would be nive to do
// Something with random variables instead.
#[derive(Clone)]
pub struct NigHyper {
    // s is distributied Normal
    pub m_mean: f64,
    pub m_std: f64,
    // s is distributied Gamma
    pub r_shape: f64,
    pub r_rate: f64,
    // s is distributied InvGamma
    pub s_shape: f64,
    pub s_rate: f64,
    // v is distributied InvGamma
    pub v_shape: f64,
    pub v_rate: f64,
}


impl Default for NigHyper {
    fn default() -> Self {
        NigHyper{m_mean: 0.0, m_std: 1.0,
                 r_shape: 1.0, r_rate: 1.0,
                 s_shape: 1.0, s_rate: 1.0,
                 v_shape: 1.0, v_rate: 1.0}
    }
}


impl NigHyper {
    pub fn new() -> Self {
        NigHyper::default()
    }

    pub fn from_data(xs: &[f64]) -> Self {
        let m = mean(xs);
        let v = var(xs);
        let s = v.sqrt();
        NigHyper{
            m_mean: m, m_std: s,
            r_shape: 2.0, r_rate: 1.0,
            s_shape: s, s_rate: 1.0/s,
            v_shape: 2.0, v_rate: 1.0}
    }

    pub fn draw(&self, mut rng: &mut Rng) -> NormalInverseGamma {
        let norm_m = Normal::new(self.m_mean, self.m_std);
        let gamma_r = Gamma::new(self.r_shape, self.r_rate);
        let gamma_s = Gamma::new(self.s_shape, self.s_rate);
        let gamma_v = Gamma::new(self.v_shape, self.v_rate);
        NormalInverseGamma{
            m: norm_m.ind_sample(&mut rng),
            r: gamma_r.draw(&mut rng),
            s: gamma_s.draw(&mut rng),
            v: gamma_v.draw(&mut rng)}
    }
}


#[cfg(test)]
mod tests {
    extern crate serde_test;

    use super::*;
    use self::serde_test::{Token, assert_tokens};

    #[test]
    fn nig_initialize() {
        let nig = NormalInverseGamma::new(1.0, 2.0, 3.0, 4.0);
        assert_relative_eq!(nig.m, 1.0, epsilon = 10E-10);
        assert_relative_eq!(nig.r, 2.0, epsilon = 10E-10);
        assert_relative_eq!(nig.s, 3.0, epsilon = 10E-10);
        assert_relative_eq!(nig.v, 4.0, epsilon = 10E-10);
    }

    #[test]
    fn nig_log_normalizer_value_1() {
        let logz = NormalInverseGamma::log_normalizer(1.0, 1.0, 1.0);
        assert_relative_eq!(logz, 1.83787706640935, epsilon = 10E-6);
    }

    #[test]
    fn nig_log_normalizer_value_2() {
        let logz = NormalInverseGamma::log_normalizer(1.2, 0.4, 5.2);
        assert_relative_eq!(logz, 5.36972819068534, epsilon = 10E-6);
    }

    #[test]
    fn nig_marginal_score_value() {
        let nig = NormalInverseGamma::new(2.1, 1.2, 1.3, 1.4);
        let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let logp = nig.marginal_score(&xs);
        assert_relative_eq!(logp, -7.69707018344038, epsilon = 10E-6);
    }

    #[test]
    fn serialize_and_deserialize() {
        let nig = NormalInverseGamma::new(0.0, 1.0, 2.0, 3.0);
        assert_tokens(&nig, &[
            Token::Struct { name: "NormalInverseGamma", len: 4 },
            Token::Str("m"),
            Token::F64(0.0),
            Token::Str("r"),
            Token::F64(1.0),
            Token::Str("s"),
            Token::F64(2.0),
            Token::Str("v"),
            Token::F64(3.0),
            Token::StructEnd,
        ]);
    }
}

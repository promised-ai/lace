extern crate rand;

use self::rand::distributions::Normal;
use self::rand::Rng;
use std::f64::consts::LN_2;

use dist::gaussian::GaussianSuffStats;
use dist::prior::Prior;
use dist::traits::{Distribution, RandomVariate, SufficientStatistic};
use dist::{Gamma, Gaussian};
use misc::mh::mh_prior;
use misc::{mean, var};
use special::gammaln;

const HALF_LOG_PI: f64 =
    0.57236494292470008193873809432261623442173004150390625;
const HALF_LOG_2PI: f64 = 0.918938533204672669540968854562379419803619384766;

// Normmal, Inverse-Gamma prior for Normal data
// --------------------------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalInverseGamma {
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
    pub hyper: NigHyper,
}

// Reference:
// https://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
impl NormalInverseGamma {
    pub fn new(m: f64, r: f64, s: f64, v: f64, hyper: NigHyper) -> Self {
        NormalInverseGamma {
            m: m,
            r: r,
            s: s,
            v: v,
            hyper: hyper,
        }
    }

    pub fn geweke() -> Self {
        NormalInverseGamma {
            m: 0.0,
            r: 1.0,
            s: 1.0,
            v: 1.0,
            hyper: NigHyper::geweke(),
        }
    }

    // TODO: implement for f32 and f64 data
    pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> Self {
        NigHyper::from_data(&xs).draw(&mut rng)
    }

    pub fn from_hyper(hyper: NigHyper, mut rng: &mut impl Rng) -> Self {
        hyper.draw(&mut rng)
    }

    fn posterior_params(&self, suffstats: &GaussianSuffStats) -> Self {
        let r = self.r + (suffstats.n as f64);
        let v = self.v + (suffstats.n as f64);
        let m = (self.m * self.r + suffstats.sum_x) / r;
        let s =
            self.s + suffstats.sum_x_sq + self.r * self.m * self.m - r * m * m;
        assert!(s > 0.0);
        NormalInverseGamma {
            m: m,
            r: r,
            s: s,
            v: v,
            hyper: self.hyper.clone(),
        }
    }

    fn log_normalizer(r: f64, s: f64, v: f64) -> f64 {
        (v + 1.0) / 2.0 * LN_2 + HALF_LOG_PI - 0.5 * r.ln() - (v / 2.0) * s.ln()
            + gammaln(v / 2.0)
    }
}

impl Prior<f64, Gaussian> for NormalInverseGamma {
    fn loglike(&self, model: &Gaussian) -> f64 {
        let rho = 1.0 / (model.sigma * model.sigma);
        let logp_rho = Gamma::new(self.v / 2.0, self.s / 2.0).loglike(&rho);
        let prior_sigma = (1.0 / (self.r * rho)).sqrt();
        let logp_mu = Gaussian::new(self.m, prior_sigma).loglike(&model.mu);
        logp_rho + logp_mu
    }

    fn posterior_draw(&self, data: &[f64], mut rng: &mut impl Rng) -> Gaussian {
        // assert!(!data.is_empty());
        let mut suffstats = GaussianSuffStats::new();
        for x in data {
            suffstats.observe(x);
        }
        assert_eq!(suffstats.n, data.len() as u64);
        self.posterior_params(&suffstats).prior_draw(&mut rng)
    }

    fn prior_draw(&self, mut rng: &mut impl Rng) -> Gaussian {
        let rho = Gamma::new(self.v / 2.0, self.s / 2.0).draw(&mut rng);
        let post_sigma = (1.0 / (self.r * rho)).sqrt();
        let mu = rng.sample(Normal::new(self.m, post_sigma));

        Gaussian::new(mu, 1.0 / rho.sqrt())
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

    fn predictive_score(&self, x: &f64, y: &[f64]) -> f64 {
        let mut suffstats = GaussianSuffStats::new();
        y.iter().for_each(|yi| suffstats.observe(yi));

        let pr_n = self.posterior_params(&suffstats);
        let zn = Self::log_normalizer(pr_n.r, pr_n.s, pr_n.v);

        suffstats.observe(&x);
        let pr_m = self.posterior_params(&suffstats);
        let zm = Self::log_normalizer(pr_m.r, pr_m.s, pr_m.v);

        -HALF_LOG_2PI + zm - zn
    }

    fn update_params<R: Rng>(
        &mut self,
        components: &[Gaussian],
        mut rng: &mut R,
    ) {
        let new_m: f64;
        let new_r: f64;
        let new_s: f64;
        let new_v: f64;

        // XXX: Can we macro these away?
        {
            let draw = |mut rng: &mut R| self.hyper.pr_m.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |m: &f64| {
                let h = self.hyper.clone();
                let nig =
                    NormalInverseGamma::new(*m, self.r, self.s, self.v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_m = mh_prior(self.m, f, draw, 50, &mut rng);
        }
        self.m = new_m;

        // update r
        {
            let draw = |mut rng: &mut R| self.hyper.pr_r.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |r: &f64| {
                let h = self.hyper.clone();
                let nig =
                    NormalInverseGamma::new(self.m, *r, self.s, self.v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_r = mh_prior(self.r, f, draw, 50, &mut rng);
        }
        self.r = new_r;

        // update s
        {
            let draw = |mut rng: &mut R| self.hyper.pr_s.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |s: &f64| {
                let h = self.hyper.clone();
                let nig =
                    NormalInverseGamma::new(self.m, self.r, *s, self.v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_s = mh_prior(self.s, f, draw, 50, &mut rng);
        }
        self.s = new_s;

        // update v
        {
            let draw = |mut rng: &mut R| self.hyper.pr_v.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |v: &f64| {
                let h = self.hyper.clone();
                let nig =
                    NormalInverseGamma::new(self.m, self.r, self.s, *v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + nig.loglike(cpnt))
            };
            new_v = mh_prior(self.v, f, draw, 50, &mut rng);
        }
        self.v = new_v;
    }
}

// Hyperprior for later?
// --------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NigHyper {
    // TODO: Change these to the correct distributions
    pub pr_m: Gaussian,
    pub pr_r: Gamma,
    pub pr_s: Gamma,
    pub pr_v: Gamma,
}

impl Default for NigHyper {
    fn default() -> Self {
        NigHyper {
            pr_m: Gaussian::new(0.0, 1.0),
            pr_r: Gamma::new(1.0, 1.0),
            pr_s: Gamma::new(1.0, 1.0),
            pr_v: Gamma::new(1.0, 1.0),
        }
    }
}

impl NigHyper {
    pub fn new(pr_m: Gaussian, pr_r: Gamma, pr_s: Gamma, pr_v: Gamma) -> Self {
        NigHyper {
            pr_m: pr_m,
            pr_r: pr_r,
            pr_s: pr_s,
            pr_v: pr_v,
        }
    }

    // FIXME: Do a better job
    /// A restrictive prior to confine Geweke
    pub fn geweke() -> Self {
        NigHyper {
            pr_m: Gaussian::new(0.0, 0.1),
            pr_r: Gamma::new(40.0, 4.0),
            pr_s: Gamma::new(40.0, 4.0),
            pr_v: Gamma::new(40.0, 4.0),
        }
    }

    pub fn from_data(xs: &[f64]) -> Self {
        let m = mean(xs);
        let v = var(xs);
        let s = v.sqrt();
        NigHyper {
            pr_m: Gaussian::new(m, s),
            pr_r: Gamma::new(2.0, 1.0),
            pr_s: Gamma::new(s, 1.0 / s),
            pr_v: Gamma::new(2.0, 1.0),
        }
    }

    pub fn draw(&self, mut rng: &mut impl Rng) -> NormalInverseGamma {
        NormalInverseGamma {
            m: self.pr_m.draw(&mut rng),
            r: self.pr_r.draw(&mut rng),
            s: self.pr_s.draw(&mut rng),
            v: self.pr_v.draw(&mut rng),
            hyper: self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate serde_test;

    use super::*;
    // use self::serde_test::{Token, assert_tokens};

    #[test]
    fn nig_initialize() {
        let hyper = NigHyper::default();
        let nig = NormalInverseGamma::new(1.0, 2.0, 3.0, 4.0, hyper);
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
        let hyper = NigHyper::default();
        let nig = NormalInverseGamma::new(2.1, 1.2, 1.3, 1.4, hyper);
        let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let logp = nig.marginal_score(&xs);
        assert_relative_eq!(logp, -7.69707018344038, epsilon = 10E-6);
    }

    #[test]
    fn nig_empty_draw_smoke_test() {
        let mut rng = rand::thread_rng();
        let hyper = NigHyper::default();
        let nig = NormalInverseGamma::new(1.0, 2.0, 3.0, 4.0, hyper);
        nig.draw(Some(&vec![]), &mut rng);
    }

    #[test]
    fn predictive_score_value_1() {
        let hyper = NigHyper::default();
        let nig = NormalInverseGamma::new(2.1, 1.2, 1.3, 1.4, hyper);
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let lp = nig.predictive_score(&3.0, &xs);
        assert_relative_eq!(lp, -1.28438638499611, epsilon = 1e-8);
    }

    #[test]
    fn predictive_score_value_2() {
        let hyper = NigHyper::default();
        let nig = NormalInverseGamma::new(2.1, 1.2, 1.3, 1.4, hyper);
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let lp = nig.predictive_score(&-3.0, &xs);
        assert_relative_eq!(lp, -6.1637698862186, epsilon = 1e-8);
    }
}

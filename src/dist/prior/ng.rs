extern crate rand;
extern crate rv;

use self::rand::Rng;

use self::rv::data::DataOrSuffStat;
use self::rv::dist::{Gamma, Gaussian, NormalGamma};
use self::rv::traits::*;
use dist::UpdatePrior;
use misc::{mean, var};
use stats::mh::mh_prior;

/// Normmal, Inverse-Gamma prior for Normal/Gassuain data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ng {
    /// Prior on parameters in N(μ, σ)
    pub ng: NormalGamma,
    /// Hyper-prior on `NormalGamma` Parameters
    pub hyper: NigHyper,
}

impl Ng {
    pub fn new(m: f64, r: f64, s: f64, v: f64, hyper: NigHyper) -> Self {
        Ng {
            ng: NormalGamma::new(m, r, s, v).unwrap(),
            hyper: hyper,
        }
    }

    /// Default prior parameters for Geweke testing
    pub fn geweke() -> Self {
        Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::geweke())
    }

    // TODO: implement for f32 and f64 data
    /// Creates an `Ng` with a vague hyper-prior derived from the data
    pub fn from_data(xs: &[f64], mut rng: &mut impl Rng) -> Self {
        NigHyper::from_data(&xs).draw(&mut rng)
    }

    /// Draws an `Ng` given a hyper-prior
    pub fn from_hyper(hyper: NigHyper, mut rng: &mut impl Rng) -> Self {
        hyper.draw(&mut rng)
    }
}

impl Rv<Gaussian> for Ng {
    fn ln_f(&self, model: &Gaussian) -> f64 {
        self.ng.ln_f(&model)
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        self.ng.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<Gaussian> {
        self.ng.sample(n, &mut rng)
    }
}

impl ConjugatePrior<f64, Gaussian> for Ng {
    type Posterior = NormalGamma;
    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> NormalGamma {
        self.ng.posterior(&x)
    }

    fn ln_m(&self, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        self.ng.ln_m(&x)
    }

    fn ln_pp(&self, y: &f64, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        self.ng.ln_pp(&y, &x)
    }
}

impl UpdatePrior<f64, Gaussian> for Ng {
    fn update_prior<R: Rng>(
        &mut self,
        components: &Vec<&Gaussian>,
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
                let ng = Ng::new(*m, self.ng.r, self.ng.s, self.ng.v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + ng.ln_f(&cpnt))
            };
            new_m = mh_prior(self.ng.m, f, draw, 50, &mut rng);
        }
        self.ng.m = new_m;

        // update r
        {
            let draw = |mut rng: &mut R| self.hyper.pr_r.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |r: &f64| {
                let h = self.hyper.clone();
                let ng = Ng::new(self.ng.m, *r, self.ng.s, self.ng.v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + ng.ln_f(&cpnt))
            };
            new_r = mh_prior(self.ng.r, f, draw, 50, &mut rng);
        }
        self.ng.r = new_r;

        // update s
        {
            let draw = |mut rng: &mut R| self.hyper.pr_s.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |s: &f64| {
                let h = self.hyper.clone();
                let ng = Ng::new(self.ng.m, self.ng.r, *s, self.ng.v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + ng.ln_f(&cpnt))
            };
            new_s = mh_prior(self.ng.s, f, draw, 50, &mut rng);
        }
        self.ng.s = new_s;

        // update v
        {
            let draw = |mut rng: &mut R| self.hyper.pr_v.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |v: &f64| {
                let h = self.hyper.clone();
                let ng = Ng::new(self.ng.m, self.ng.r, self.ng.s, *v, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + ng.ln_f(&cpnt))
            };
            new_v = mh_prior(self.ng.v, f, draw, 50, &mut rng);
        }
        self.ng.v = new_v;
    }
}

/// Hyper-prior for Normal Gamma (`Ng`)
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NigHyper {
    // TODO: Change these to the correct distributions, according to the
    // rasmussen IGMM paper
    /// Prior on `m`
    pub pr_m: Gaussian,
    /// Prior on `r`
    pub pr_r: Gamma,
    /// Prior on `s`
    pub pr_s: Gamma,
    /// Prior on `v`
    pub pr_v: Gamma,
}

impl Default for NigHyper {
    fn default() -> Self {
        NigHyper {
            pr_m: Gaussian::new(0.0, 1.0).unwrap(),
            pr_r: Gamma::new(1.0, 1.0).unwrap(),
            pr_s: Gamma::new(1.0, 1.0).unwrap(),
            pr_v: Gamma::new(1.0, 1.0).unwrap(),
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

    /// A restrictive prior to confine Geweke.
    ///
    /// Since the geweke test seeks to draw samples from the joint of the prior
    /// and the data, p(x, θ), and since θ is indluenced by the hyper-prior, if
    /// the hyper parameters are not tight, the data can go crazy and cause a
    /// bunch of math errors.
    pub fn geweke() -> Self {
        NigHyper {
            pr_m: Gaussian::new(0.0, 0.1).unwrap(),
            pr_r: Gamma::new(40.0, 4.0).unwrap(),
            pr_s: Gamma::new(40.0, 4.0).unwrap(),
            pr_v: Gamma::new(40.0, 4.0).unwrap(),
        }
    }

    /// Vague prior from the data.
    pub fn from_data(xs: &[f64]) -> Self {
        let m = mean(xs);
        let v = var(xs);
        let s = v.sqrt();
        NigHyper {
            pr_m: Gaussian::new(m, s).unwrap(),
            pr_r: Gamma::new(2.0, 1.0).unwrap(),
            pr_s: Gamma::new(s, 1.0 / s).unwrap(),
            pr_v: Gamma::new(2.0, 1.0).unwrap(),
        }
    }

    /// Draw an `Ng` from the hyper
    pub fn draw(&self, mut rng: &mut impl Rng) -> Ng {
        Ng::new(
            self.pr_m.draw(&mut rng),
            self.pr_r.draw(&mut rng),
            self.pr_s.draw(&mut rng),
            self.pr_v.draw(&mut rng),
            self.clone(),
        )
    }
}

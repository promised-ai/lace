use crate::rv::data::DataOrSuffStat;
use crate::rv::dist::{
    Categorical, CategoricalError, Gamma, SymmetricDirichlet,
};
use crate::rv::traits::{
    ConjugatePrior, Entropy, HasSuffStat, Mode, Rv, SuffStat,
};
use crate::UpdatePrior;
use lace_consts::rv::data::CategoricalSuffStat;
use lace_consts::rv::prelude::{Beta, Dirichlet};
use serde::{Deserialize, Serialize};

/// The minimum stick breaking mass
pub const BETA_0: f64 = 1e-6;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DpdSuffStat {
    n: usize,
    counts: Vec<f64>,
}

impl DpdSuffStat {
    pub fn new() -> Self {
        Self {
            n: 0,
            counts: Vec::new(),
        }
    }
}

impl SuffStat<usize> for DpdSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &usize) {
        if *x >= self.counts.len() {
            self.counts.resize(*x + 1, 0.0);
        }

        self.n += 1;
        self.counts[*x] += 1.0;
    }

    fn forget(&mut self, x: &usize) {
        self.n -= 1;
        self.counts[*x] -= 1.0;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DpdHyper {
    pub gamma: Gamma,
}

impl Default for DpdHyper {
    fn default() -> Self {
        Self {
            gamma: Gamma::new_unchecked(2.0, 2.0),
        }
    }
}

impl DpdHyper {
    pub fn draw<R: rand::Rng>(
        &self,
        k_r: usize,
        m: usize,
        rng: &mut R,
    ) -> DpdPrior {
        let alpha = self.gamma.draw(rng);
        DpdPrior::Symmetric {
            k_r,
            m,
            dir: SymmetricDirichlet::new_unchecked(alpha, k_r + m),
        }
    }
}

impl DpdHyper {
    pub fn new() -> Self {
        DpdHyper::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Dpd {
    k: usize,
    m: usize,
    categorical: Categorical,
}

impl Dpd {
    pub fn new(k: usize, weights: &[f64]) -> Result<Self, CategoricalError> {
        let categorical = Categorical::new(weights)?;
        Ok(Self {
            k,
            m: categorical.k() - k,
            categorical,
        })
    }

    pub fn uniform(k: usize, m: usize) -> Self {
        Self {
            k,
            m,
            categorical: Categorical::uniform(k + m),
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn ln_weights(&self) -> &[f64] {
        self.categorical.ln_weights()
    }

    pub fn len(&self) -> usize {
        self.k + self.m
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Rv<usize> for Dpd {
    fn ln_f(&self, x: &usize) -> f64 {
        let ln_weights = self.categorical.ln_weights();
        if *x >= self.k {
            ln_weights[self.k]
        } else {
            ln_weights[*x]
        }
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.categorical.draw(rng)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DpdPrior {
    Symmetric {
        k_r: usize,
        m: usize,
        dir: SymmetricDirichlet,
    },
    Dirichlet {
        k_r: usize,
        m: usize,
        dir: Dirichlet,
    },
}

impl DpdPrior {
    pub fn new_unchecked(k_realized: usize, m: usize, alpha: f64) -> Self {
        Self::Symmetric {
            k_r: k_realized,
            m,
            dir: SymmetricDirichlet::new_unchecked(alpha, k_realized + m),
        }
    }

    pub fn sym_alpha(&self) -> Option<f64> {
        match self {
            Self::Symmetric { dir, .. } => Some(dir.alpha()),
            _ => None,
        }
    }

    fn set_sym_alpha(&mut self, alpha: f64) {
        match self {
            Self::Symmetric { dir, .. } => dir.set_alpha(alpha).unwrap(),
            _ => {
                panic!("cannot set sym_alpha of Dirichlet variant")
            }
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Self::Symmetric { k_r, .. } => *k_r,
            Self::Dirichlet { k_r, .. } => *k_r,
        }
    }

    pub fn m(&self) -> usize {
        match self {
            Self::Symmetric { m, .. } => *m,
            Self::Dirichlet { m, .. } => *m,
        }
    }
}

impl HasSuffStat<usize> for Dpd {
    type Stat = DpdSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

    fn ln_f_stat(&self, _stat: &Self::Stat) -> f64 {
        unimplemented!()
    }
}

impl Rv<Dpd> for DpdPrior {
    fn ln_f(&self, x: &Dpd) -> f64 {
        let beta = Beta::new_unchecked(1.0, self.sym_alpha().unwrap());
        x.categorical
            .ln_weights()
            .iter()
            .fold((1.0, 0.0), |(b0, ln_f), &ln_w| {
                let w = ln_w.exp();
                (b0 - w, ln_f + beta.ln_f(&(w / b0)))
            })
            .1
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Dpd {
        match self {
            Self::Symmetric { k_r, m, dir } => {
                let cat: Categorical = dir.draw(rng);
                Dpd {
                    k: *k_r,
                    m: *m,
                    categorical: cat,
                }
            }
            Self::Dirichlet { k_r, m, dir } => {
                let cat: Categorical = dir.draw(rng);
                Dpd {
                    k: *k_r,
                    m: *m,
                    categorical: cat,
                }
            }
        }
    }
}

impl Rv<Dpd> for SymmetricDirichlet {
    fn ln_f(&self, x: &Dpd) -> f64 {
        self.ln_f(&x.categorical)
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Dpd {
        let categorical: Categorical = self.draw(rng);
        Dpd {
            k: categorical.k(),
            m: 3,
            categorical,
        }
    }
}

impl Rv<Dpd> for Dirichlet {
    fn ln_f(&self, x: &Dpd) -> f64 {
        self.ln_f(&x.categorical)
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Dpd {
        let categorical: Categorical = self.draw(rng);
        Dpd {
            k: categorical.k(),
            m: 3,
            categorical,
        }
    }
}

fn dpd_posterior_from_stat(
    stat: &CategoricalSuffStat,
    dir: &SymmetricDirichlet,
) -> Dirichlet {
    let k = stat.counts().len();
    let alpha = dir.alpha();
    let stat = DataOrSuffStat::SuffStat(stat);

    if dir.k() < k {
        let symdir = SymmetricDirichlet::new(alpha, k).unwrap();
        <SymmetricDirichlet as ConjugatePrior<usize, Categorical>>::posterior(
            &symdir, &stat,
        )
    } else if dir.k() > k {
        let symdir = SymmetricDirichlet::new(alpha, dir.k()).unwrap();
        <SymmetricDirichlet as ConjugatePrior<usize, Categorical>>::posterior(
            &symdir, &stat,
        )
    } else {
        <SymmetricDirichlet as ConjugatePrior<usize, Categorical>>::posterior(
            dir, &stat,
        )
    }
}

fn dpd_to_cat_stat(x: &DataOrSuffStat<usize, Dpd>) -> CategoricalSuffStat {
    match x {
        DataOrSuffStat::Data(xs) => {
            let k = xs.iter().max().unwrap() + 1;
            let mut stat_cat = CategoricalSuffStat::new(k);
            stat_cat.observe_many(&xs);
            stat_cat
        }
        DataOrSuffStat::SuffStat(stat) => {
            let k = stat.counts.len();
            let mut stat_cat = CategoricalSuffStat::new(k);
            for (ix, ct) in stat.counts.iter().enumerate() {
                for _ in 0..(ct.round() as usize) {
                    stat_cat.observe(&ix)
                }
            }
            stat_cat
        }
        DataOrSuffStat::None => {
            panic!("Should never have `None` Stat");
        }
    }
}

impl ConjugatePrior<usize, Dpd> for DpdPrior {
    type LnMCache = ();
    type LnPpCache = ();
    type Posterior = DpdPrior;

    fn ln_m_cache(&self) -> Self::LnMCache {}

    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Dpd>) -> Self::LnPpCache {}

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
        panic!("Cannot compute ln_pp_with_cache for Dpd")
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::LnMCache,
        x: &DataOrSuffStat<usize, Dpd>,
    ) -> f64 {
        panic!("Cannot compute ln_m_with_cache for Dpd")
    }

    fn posterior(&self, x: &DataOrSuffStat<usize, Dpd>) -> Self::Posterior {
        if let Self::Symmetric { dir, m, k_r } = self {
            let stat = dpd_to_cat_stat(x);
            let posterior = dpd_posterior_from_stat(&stat, &dir);
            DpdPrior::Dirichlet {
                m: *m,
                k_r: *k_r,
                dir: posterior,
            }
        } else {
            panic!("Cannot compute posterior for DpdPrior::Dirichlet")
        }
    }
}

impl Entropy for Dpd {
    fn entropy(&self) -> f64 {
        self.categorical.entropy()
    }
}

impl Mode<usize> for Dpd {
    fn mode(&self) -> Option<usize> {
        self.categorical.mode()
    }
}

impl UpdatePrior<usize, Dpd, DpdHyper> for DpdPrior {
    fn update_prior<R: rand::Rng>(
        &mut self,
        components: &[&Dpd],
        hyper: &DpdHyper,
        rng: &mut R,
    ) -> f64 {
        let k = components[0].k;

        let ln_prior = |alpha: &f64| {
            let dir = SymmetricDirichlet::new_unchecked(*alpha, k);
            components
                .iter()
                .map(|cpnt| dir.ln_f(&cpnt.categorical))
                .sum::<f64>()
        };

        let res = crate::mh::mh_prior(
            self.sym_alpha().unwrap(),
            ln_prior,
            |r| hyper.gamma.draw(r),
            200,
            rng,
        );

        self.set_sym_alpha(res.x);

        res.score_x
    }
}

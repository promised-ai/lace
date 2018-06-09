extern crate num;
extern crate rand;
extern crate serde;
extern crate serde_yaml;

use std::marker::Sync;

use self::rand::Rng;
use self::serde::Serialize;

use cc::assignment::Assignment;
use cc::container::DataContainer;
use dist::prior::Prior;
use dist::traits::{AccumScore, Distribution};

use std::collections::BTreeMap;
use geweke::traits::*;
use misc::{mean, std};
use dist::{Gaussian, Categorical};
use dist::prior::{NormalInverseGamma, CatSymDirichlet};
use dist::prior::nig::NigHyper;
use dist::prior::csd::CsdHyper;
use dist::traits::RandomVariate;

#[derive(Serialize, Deserialize, Clone)]
pub struct Column<T, M, R>
where
    T: Clone + Sync,
    M: Distribution<T> + AccumScore<T> + Serialize,
    R: Prior<T, M> + Serialize,
{
    pub id: usize,
    // TODO: Fiure out a way to optionally serialize data
    pub data: DataContainer<T>,
    pub components: Vec<M>,
    pub prior: R,
    // TODO: pointers to data on GPU
}

impl<T, M, R> Column<T, M, R>
where
    T: Clone + Sync,
    M: Distribution<T> + AccumScore<T> + Serialize,
    R: Prior<T, M> + Serialize,
{
    pub fn new(id: usize, data: DataContainer<T>, prior: R) -> Self {
        Column {
            id: id,
            data: data,
            components: Vec::new(),
            prior: prior,
        }
    }

    pub fn len(&self) -> usize {
        // XXX: this will fail on features with dropped data
        self.data.len()
    }
}

pub trait Feature {
    fn id(&self) -> usize;
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize);
    fn init_components(&mut self, k: usize, rng: &mut impl Rng);
    fn update_components(&mut self, asgn: &Assignment, rng: &mut impl Rng);
    fn reassign(&mut self, asgn: &Assignment, rng: &mut impl Rng);
    fn col_score(&self, asgn: &Assignment) -> f64;
    fn update_prior_params(&mut self, rng: &mut impl Rng);
    fn append_empty_component(&mut self, rng: &mut impl Rng);
    fn drop_component(&mut self, k: usize);
    fn len(&self) -> usize;
    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64>;

    // fn yaml(&self) -> String;
    // fn geweke_resample_data(&mut self, asgn: &Assignment, &mut Rng);
}

#[allow(dead_code)]
impl<T, M, R> Feature for Column<T, M, R>
where
    M: Distribution<T> + AccumScore<T> + Serialize,
    T: Clone + Sync,
    R: Prior<T, M> + Serialize,
{
    fn id(&self) -> usize {
        self.id
    }

    fn accum_score(&self, mut scores: &mut Vec<f64>, k: usize) {
        // TODO: Decide when to use parallel or GPU
        self.components[k].accum_score(
            &mut scores,
            &self.data.data,
            &self.data.present,
        );
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn init_components(&mut self, k: usize, mut rng: &mut impl Rng) {
        self.components =
            (0..k).map(|_| self.prior.prior_draw(&mut rng)).collect()
    }

    fn update_components(&mut self, asgn: &Assignment, mut rng: &mut impl Rng) {
        self.components = self
            .data
            .group_by(asgn)
            .iter()
            .map(|xk| self.prior.draw(Some(&xk), &mut rng))
            .collect()
    }

    fn reassign(&mut self, asgn: &Assignment, mut rng: &mut impl Rng) {
        self.update_components(&asgn, &mut rng);
    }

    fn col_score(&self, asgn: &Assignment) -> f64 {
        self.data
            .group_by(asgn)
            .iter()
            .fold(0.0, |acc, xk| acc + self.prior.marginal_score(xk))
    }

    fn update_prior_params(&mut self, mut rng: &mut impl Rng) {
        self.prior.update_params(&self.components, &mut rng);
    }

    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
        self.components.push(self.prior.draw(None, &mut rng));
    }

    fn drop_component(&mut self, k: usize) {
        // cpnt goes out of scope and is dropped (Hopefully)
        let _cpnt = self.components.remove(k);
    }

    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64> {
        if self.data.present[row_ix] {
            let x = &self.data.data[row_ix];
            let cpnt = &self.components[k];
            Some(cpnt.loglike(x))
        } else {
            None
        }
    }

    // fn yaml(&self) -> String {
    //     serde_yaml::to_string(&self).unwrap()
    // }

    // // XXX: One day this should go away. See comment in Feature definition.
    // fn geweke_resample_data(&mut self, asgn: &Assignment, rng: &mut Rng) {
    //     for (i, &k) in asgn.asgn.iter().enumerate() {
    //         self.data[i] = self.components[k].draw(rng);
    //     }
    // }
}

// Geweke implementations
// ======================

// Continuous
// ----------
impl GewekeModel for Column<f64, Gaussian, NormalInverseGamma> {
    fn geweke_from_prior(settings: &Assignment, mut rng: &mut impl Rng) -> Self {
        let f = Gaussian::new(0.0, 1.0);
        let xs = f.sample(settings.len(), &mut rng);
        let data = DataContainer::new(xs); // initial data is resampled anyway
        let prior = NigHyper::geweke().draw(&mut rng);
        let mut col = Column::new(0, data, prior);
        col.init_components(settings.ncats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(&mut self, settings: &Assignment, mut rng: &mut impl Rng) {
        self.update_components(settings, &mut rng);
        self.update_prior_params(&mut rng);
    }
}

impl GewekeResampleData for Column<f64, Gaussian, NormalInverseGamma> {
    type Settings = Assignment;
    fn geweke_resample_data(&mut self,
        s: Option<&Self::Settings>,
        rng: &mut impl Rng,
    ) {
        let asgn = s.unwrap();
        for (i, &k) in asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }
}

impl GewekeSummarize for Column<f64, Gaussian, NormalInverseGamma> {
    fn geweke_summarize(&self) -> BTreeMap<String, f64> {
        let x_mean = mean(&self.data.data);
        let x_std = std(&self.data.data);

        let mus: Vec<f64> = self.components
            .iter()
            .map(|c| c.mu)
            .collect();

        let sigmas: Vec<f64> = self.components
            .iter()
            .map(|c| c.sigma)
            .collect();

        let mu_mean = mean(&mus);
        let sigma_mean = mean(&sigmas);

        let mut stats: BTreeMap<String, f64> = BTreeMap::new();

        stats.insert(String::from("x mean"), x_mean);
        stats.insert(String::from("x std"), x_std);
        stats.insert(String::from("mu mean"), mu_mean);
        stats.insert(String::from("sigma mean"), sigma_mean);

        stats
    }
}

// Categorical
// -----------
impl GewekeModel for Column<u8, Categorical<u8>, CatSymDirichlet> {
    fn geweke_from_prior(settings: &Assignment, mut rng: &mut impl Rng) -> Self {
        let k = 5;
        let f = Categorical::flat(k);
        let xs = f.sample(settings.len(), &mut rng);
        let data = DataContainer::new(xs); // initial data is resampled anyway
        let prior = CsdHyper::geweke().draw(k, &mut rng);
        // let prior = CatSymDirichlet::new(1.0, k, CsdHyper::geweke());
        let mut col = Column::new(0, data, prior);
        col.init_components(settings.ncats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(&mut self, settings: &Assignment, mut rng: &mut impl Rng) {
        self.update_components(settings, &mut rng);
        self.update_prior_params(&mut rng);
    }
}

// TODO: Make a macro for this
impl GewekeResampleData for Column<u8, Categorical<u8>, CatSymDirichlet> {
    type Settings = Assignment;
    fn geweke_resample_data(&mut self,
        s: Option<&Self::Settings>,
        rng: &mut impl Rng,
    ) {
        let asgn = s.unwrap();
        for (i, &k) in asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }
}

impl GewekeSummarize for Column<u8, Categorical<u8>, CatSymDirichlet> {
    fn geweke_summarize(&self) -> BTreeMap<String, f64> {
        let x_sum = self.data.data.iter().fold(0, |acc, x| acc + x);

        fn sum_sq(logws: &[f64]) -> f64 {
            logws.iter().fold(0.0, |acc, lw| {
                let w = lw.exp();
                acc + w * w
            })
        }

        let k = self.components.len() as f64;
        let mean_hrm: f64 =
            self.components
                .iter()
                .fold(0.0, |acc, cpnt| acc + sum_sq(&cpnt.log_weights)) / k;

        let mut stats: BTreeMap<String, f64> = BTreeMap::new();

        stats.insert(String::from("x sum"), x_sum as f64);
        stats.insert(String::from("weight sum squares"), mean_hrm as f64);

        stats
    }
}

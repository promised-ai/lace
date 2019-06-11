//! Geweke implementations
use braid_stats::prior::{Csd, CsdHyper, Ng, NigHyper};
use braid_utils::stats::{mean, std};
use rv::{
    dist::{Categorical, Gaussian},
    traits::Rv,
};
use std::collections::BTreeMap;

use rand::Rng;

use super::ColModel;
use crate::{
    cc::{
        transition::ViewTransition, Assignment, Column, DataContainer, FType,
        Feature,
    },
    geweke::{GewekeModel, GewekeResampleData, GewekeSummarize},
};

#[derive(Clone)]
pub struct ColumnGewekeSettings {
    asgn: Assignment,
    transitions: Vec<ViewTransition>,
    fixed_prior: bool,
}

impl ColumnGewekeSettings {
    pub fn new(asgn: Assignment, transitions: Vec<ViewTransition>) -> Self {
        let fixed_prior = transitions
            .iter()
            .find(|t| **t == ViewTransition::FeaturePriors)
            .is_none();

        ColumnGewekeSettings {
            asgn: asgn,
            transitions: transitions,
            fixed_prior: fixed_prior,
        }
    }
}

// Continuous
// ----------
impl GewekeModel for Column<f64, Gaussian, Ng> {
    fn geweke_from_prior(
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) -> Self {
        let f = Gaussian::new(0.0, 1.0).unwrap();
        let xs = f.sample(settings.asgn.len(), &mut rng);
        let data = DataContainer::new(xs); // initial data is re-sampled anyway
        let prior = if settings.fixed_prior {
            Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::geweke())
        } else {
            NigHyper::geweke().draw(&mut rng)
        };
        let mut col = Column::new(0, data, prior);
        col.init_components(settings.asgn.ncats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(
        &mut self,
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) {
        self.update_components(&mut rng);
        if !settings.fixed_prior {
            self.update_prior_params(&mut rng);
        }
    }
}

impl GewekeResampleData for Column<f64, Gaussian, Ng> {
    type Settings = ColumnGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&Self::Settings>,
        rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        for (i, &k) in s.asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }
}

impl GewekeSummarize for Column<f64, Gaussian, Ng> {
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> BTreeMap<String, f64> {
        let x_mean = mean(&self.data.data);
        let x_std = std(&self.data.data);

        let mus: Vec<f64> = self.components.iter().map(|c| c.fx.mu).collect();

        let sigmas: Vec<f64> =
            self.components.iter().map(|c| c.fx.sigma).collect();

        let mu_mean = mean(&mus);
        let sigma_mean = mean(&sigmas);

        let mut stats: BTreeMap<String, f64> = BTreeMap::new();

        stats.insert(String::from("x mean"), x_mean);
        stats.insert(String::from("x std"), x_std);
        stats.insert(String::from("mu mean"), mu_mean);
        stats.insert(String::from("sigma mean"), sigma_mean);
        if !settings.fixed_prior {
            stats.insert(String::from("NIG m"), self.prior.ng.m);
            stats.insert(String::from("NIG r"), self.prior.ng.r);
            stats.insert(String::from("NIG s"), self.prior.ng.s);
            stats.insert(String::from("NIG v"), self.prior.ng.v);
        }

        stats
    }
}

// Categorical
// -----------
impl GewekeModel for Column<u8, Categorical, Csd> {
    fn geweke_from_prior(
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) -> Self {
        let k = 5;
        let f = Categorical::uniform(k);
        let xs = f.sample(settings.asgn.len(), &mut rng);
        let data = DataContainer::new(xs); // initial data is resampled anyway
        let prior = if settings.fixed_prior {
            Csd::new(1.0, k, CsdHyper::geweke())
        } else {
            CsdHyper::geweke().draw(k, &mut rng)
        };
        let mut col = Column::new(0, data, prior);
        col.init_components(settings.asgn.ncats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(
        &mut self,
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) {
        self.update_components(&mut rng);
        if !settings.fixed_prior {
            self.update_prior_params(&mut rng);
        }
    }
}

// TODO: Make a macro for this
impl GewekeResampleData for Column<u8, Categorical, Csd> {
    type Settings = ColumnGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&Self::Settings>,
        rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        for (i, &k) in s.asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }
}

impl GewekeSummarize for Column<u8, Categorical, Csd> {
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> BTreeMap<String, f64> {
        let x_sum = self.data.data.iter().fold(0, |acc, x| acc + x);

        fn sum_sq(logws: &[f64]) -> f64 {
            logws.iter().fold(0.0, |acc, lw| acc + lw.exp().powi(2))
        }

        fn weight_mean(logws: &[f64]) -> f64 {
            let k = logws.len() as f64;
            logws.iter().fold(0.0, |acc, lw| acc + lw) / k
        }

        let k = self.components.len() as f64;
        let mean_hrm: f64 = self
            .components
            .iter()
            .fold(0.0, |acc, cpnt| acc + sum_sq(&cpnt.fx.ln_weights))
            / k;

        let mean_weight: f64 = self
            .components
            .iter()
            .fold(0.0, |acc, cpnt| acc + weight_mean(&cpnt.fx.ln_weights))
            / k;

        let mut stats: BTreeMap<String, f64> = BTreeMap::new();

        stats.insert(String::from("x sum"), x_sum as f64);
        stats.insert(String::from("weight sum squares"), mean_hrm as f64);
        stats.insert(String::from("weight mean"), mean_weight as f64);
        if !settings.fixed_prior {
            stats.insert(String::from("prior alpha"), self.prior.symdir.alpha);
        }

        stats
    }
}

// ColumnModel
// ===========
impl GewekeSummarize for ColModel {
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> BTreeMap<String, f64> {
        match *self {
            ColModel::Continuous(ref f) => f.geweke_summarize(&settings),
            ColModel::Categorical(ref f) => f.geweke_summarize(&settings),
            _ => unimplemented!("Unsupported column type"),
        }
    }
}

impl GewekeResampleData for ColModel {
    type Settings = ColumnGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&Self::Settings>,
        mut rng: &mut impl Rng,
    ) {
        match *self {
            ColModel::Continuous(ref mut f) => {
                f.geweke_resample_data(settings, &mut rng)
            }
            ColModel::Categorical(ref mut f) => {
                f.geweke_resample_data(settings, &mut rng)
            }
            _ => unimplemented!("Unsupported column type"),
        }
    }
}

pub fn gen_geweke_col_models(
    cm_types: &[FType],
    nrows: usize,
    do_ftr_prior_transition: bool,
    mut rng: &mut impl Rng,
) -> Vec<ColModel> {
    cm_types
        .iter()
        .enumerate()
        .map(|(id, cm_type)| {
            match cm_type {
                FType::Continuous => {
                    let prior = if do_ftr_prior_transition {
                        NigHyper::geweke().draw(&mut rng)
                    } else {
                        Ng::geweke()
                    };
                    // This is filler data, it SHOULD be overwritten at the
                    // start of the geweke run
                    let f = prior.draw(&mut rng);
                    let xs = f.sample(nrows, &mut rng);
                    let data = DataContainer::new(xs);
                    let column = Column::new(id, data, prior);
                    ColModel::Continuous(column)
                }
                FType::Categorical => {
                    let k = 5; // number of categorical values
                    let prior = if do_ftr_prior_transition {
                        CsdHyper::geweke().draw(k, &mut rng)
                    } else {
                        Csd::geweke(k)
                    };
                    let f = prior.draw(&mut rng);
                    let xs = f.sample(nrows, &mut rng);
                    let data = DataContainer::new(xs);
                    let column = Column::new(id, data, prior);
                    ColModel::Categorical(column)
                }
                _ => unimplemented!("Unsupported FType"),
            }
        })
        .collect()
}

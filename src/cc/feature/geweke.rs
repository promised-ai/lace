//! Geweke implementations
use std::collections::BTreeMap;

use braid_stats::prior::{Csd, CsdHyper, Ng, NigHyper, Pg, PgHyper};
use braid_utils::{mean, std};
use rand::Rng;
use rv::dist::{Categorical, Gaussian, Poisson};
use rv::traits::Rv;

use super::ColModel;
use crate::cc::transition::ViewTransition;
use crate::cc::{Assignment, Column, DataContainer, FType, Feature};
use braid_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};

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
            asgn,
            transitions,
            fixed_prior,
        }
    }
}

/// Summary for the Normal Gamma prior. Just a list of parameters
#[derive(Clone, Debug)]
pub struct NgSummary {
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
}

#[derive(Clone, Debug)]
pub struct PgSummary {
    pub shape: f64,
    pub rate: f64,
}

/// Column summary for Geweke
#[derive(Clone, Debug)]
pub enum GewekeColumnSummary {
    /// Continuous feature
    Continuous {
        /// data mean
        x_mean: f64,
        /// data standard deviation
        x_std: f64,
        /// Mean of the component mu parameters
        mu_mean: f64,
        /// Mean of the component sigma parameters
        sigma_mean: f64,
        /// The prior parameter summary if the prior was variable
        ng: Option<NgSummary>,
    },
    /// Categorical feature
    Categorical {
        /// The sum of the data
        x_sum: u32,
        /// The mean of the squared weights
        sq_weight_mean: f64,
        /// The mean of the weights
        weight_mean: f64,
        /// The prior alpha summary if the prior was variable
        prior_alpha: Option<f64>,
    },
    /// Count feature
    Count {
        /// The sum of the data
        x_sum: u32,
        /// Mean of the data
        x_mean: f64,
        /// Mean of the component rate parameters
        rate_mean: f64,
        /// The prior parameter summary if the prior was variable
        pg: Option<PgSummary>,
    },
}

impl From<&GewekeColumnSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeColumnSummary) -> Self {
        match value {
            GewekeColumnSummary::Continuous {
                x_mean,
                x_std,
                mu_mean,
                sigma_mean,
                ng,
            } => {
                let mut map: BTreeMap<String, f64> = BTreeMap::new();
                map.insert("x mean".into(), *x_mean);
                map.insert("x std".into(), *x_std);
                map.insert("mu mean".into(), *mu_mean);
                map.insert("sigma mean".into(), *sigma_mean);
                if let Some(inner) = ng {
                    map.insert("ng m".into(), inner.m);
                    map.insert("ng r".into(), inner.r);
                    map.insert("ng s".into(), inner.s);
                    map.insert("ng v".into(), inner.v);
                }
                map
            }
            GewekeColumnSummary::Categorical {
                x_sum,
                sq_weight_mean,
                weight_mean,
                prior_alpha,
            } => {
                let mut map: BTreeMap<String, f64> = BTreeMap::new();
                map.insert("x sum".into(), *x_sum as f64);
                map.insert("sq weight mean".into(), *sq_weight_mean);
                map.insert("weight mean".into(), *weight_mean);
                if let Some(alpha) = prior_alpha {
                    map.insert("csd alpha".into(), *alpha);
                }
                map
            }
            GewekeColumnSummary::Count {
                x_sum,
                x_mean,
                rate_mean,
                pg,
            } => {
                let mut map: BTreeMap<String, f64> = BTreeMap::new();
                map.insert("x sum".into(), f64::from(*x_sum));
                map.insert("x mean".into(), *x_mean);
                map.insert("rate mean".into(), *rate_mean);
                if let Some(inner) = pg {
                    map.insert("pg shape".into(), inner.shape);
                    map.insert("pg rate".into(), inner.rate);
                }
                map
            }
        }
    }
}

impl From<GewekeColumnSummary> for BTreeMap<String, f64> {
    fn from(value: GewekeColumnSummary) -> Self {
        Self::from(&value)
    }
}

macro_rules! impl_gewek_resample {
    ($x:ty, $fx:ty, $pr:ty) => {
        // TODO: Make a macro for this
        impl GewekeResampleData for Column<$x, $fx, $pr> {
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
    };
}

impl_gewek_resample!(u8, Categorical, Csd);
impl_gewek_resample!(f64, Gaussian, Ng);
impl_gewek_resample!(u32, Poisson, Pg);

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

impl GewekeSummarize for Column<f64, Gaussian, Ng> {
    type Summary = GewekeColumnSummary;

    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> Self::Summary {
        let x_mean = mean(&self.data.data);
        let x_std = std(&self.data.data);

        let mus: Vec<f64> = self.components.iter().map(|c| c.fx.mu()).collect();

        let sigmas: Vec<f64> =
            self.components.iter().map(|c| c.fx.sigma()).collect();

        let mu_mean = mean(&mus);
        let sigma_mean = mean(&sigmas);

        GewekeColumnSummary::Continuous {
            x_mean,
            x_std,
            mu_mean,
            sigma_mean,
            ng: if !settings.fixed_prior {
                Some(NgSummary {
                    m: self.prior.ng.m(),
                    r: self.prior.ng.r(),
                    s: self.prior.ng.s(),
                    v: self.prior.ng.v(),
                })
            } else {
                None
            },
        }
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

impl GewekeSummarize for Column<u8, Categorical, Csd> {
    type Summary = GewekeColumnSummary;
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> Self::Summary {
        let x_sum = self
            .data
            .data
            .iter()
            .fold(0_u32, |acc, &x| acc + u32::from(x));

        fn sum_sq(logws: &[f64]) -> f64 {
            logws.iter().fold(0.0, |acc, lw| acc + lw.exp().powi(2))
        }

        let k = self.components.len() as f64;
        let sq_weight_mean: f64 = self
            .components
            .iter()
            .fold(0.0, |acc, cpnt| acc + sum_sq(&cpnt.fx.ln_weights()))
            / k;

        let weight_mean: f64 = self.components.iter().fold(0.0, |acc, cpnt| {
            let kw = cpnt.fx.ln_weights().len() as f64;
            let mean =
                cpnt.fx.ln_weights().iter().fold(0.0, |acc, lw| acc + lw) / kw;
            acc + mean
        }) / k;

        GewekeColumnSummary::Categorical {
            x_sum,
            sq_weight_mean,
            weight_mean,
            prior_alpha: if !settings.fixed_prior {
                Some(self.prior.symdir.alpha())
            } else {
                None
            },
        }
    }
}

// Count model
// -----------
impl GewekeSummarize for Column<u32, Poisson, Pg> {
    type Summary = GewekeColumnSummary;

    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> Self::Summary {
        let x_sum = self.data.data.iter().sum::<u32>();

        let x_mean = f64::from(self.data.data.iter().sum::<u32>())
            / self.data.len() as f64;

        let rate_mean =
            self.components.iter().map(|c| c.fx.rate()).sum::<f64>()
                / self.components.len() as f64;

        GewekeColumnSummary::Count {
            x_sum,
            x_mean,
            rate_mean,
            pg: if !settings.fixed_prior {
                Some(PgSummary {
                    shape: self.prior.gamma.shape(),
                    rate: self.prior.gamma.rate(),
                })
            } else {
                None
            },
        }
    }
}

// ColumnModel
// ===========
impl GewekeSummarize for ColModel {
    type Summary = GewekeColumnSummary;
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> Self::Summary {
        match *self {
            ColModel::Continuous(ref f) => f.geweke_summarize(&settings),
            ColModel::Categorical(ref f) => f.geweke_summarize(&settings),
            ColModel::Count(ref f) => f.geweke_summarize(&settings),
            ColModel::Labeler(..) => panic!("Unsupported col type"),
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
            ColModel::Count(ref mut f) => {
                f.geweke_resample_data(settings, &mut rng)
            }
            _ => unimplemented!("Unsupported column type"),
        }
    }
}

macro_rules! geweke_cm_arm {
    (
        $prior_trans: ident,
        $rng: ident,
        $id: ident,
        $nrows: ident,
        $prior_type: ty,
        $hyper_type: ty,
        $cmvar: ident
    ) => {{
        let prior = if $prior_trans {
            <$hyper_type>::geweke().draw(&mut $rng)
        } else {
            <$prior_type>::geweke()
        };
        // This is filler data, it SHOULD be overwritten at the
        // start of the geweke run
        let f = prior.draw(&mut $rng);
        let xs = f.sample($nrows, &mut $rng);
        let data = DataContainer::new(xs);
        let column = Column::new($id, data, prior);
        ColModel::$cmvar(column)
    }};
}

pub fn gen_geweke_col_models(
    cm_types: &[FType],
    nrows: usize,
    prior_trans: bool,
    mut rng: &mut impl Rng,
) -> Vec<ColModel> {
    cm_types
        .iter()
        .enumerate()
        .map(|(id, cm_type)| {
            match cm_type {
                FType::Continuous => geweke_cm_arm!(
                    prior_trans,
                    rng,
                    id,
                    nrows,
                    Ng,
                    NigHyper,
                    Continuous
                ),
                FType::Count => geweke_cm_arm!(
                    prior_trans,
                    rng,
                    id,
                    nrows,
                    Pg,
                    PgHyper,
                    Count
                ),
                FType::Categorical => {
                    let k = 5; // number of categorical values
                    let prior = if prior_trans {
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

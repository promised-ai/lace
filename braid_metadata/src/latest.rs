use std::collections::BTreeMap;

use braid_cc::assignment::Assignment;
use braid_cc::component::ConjugateComponent;
use braid_cc::feature::{ColModel, Column};
use braid_cc::state::{State, StateDiagnostics};
use braid_cc::traits::{BraidDatum, BraidLikelihood, BraidPrior, BraidStat};
use braid_cc::view::View;
use braid_codebook::Codebook;
use braid_data::label::Label;
use braid_data::DataStore;
use braid_data::SparseContainer;
use braid_stats::labeler::{Labeler, LabelerPrior};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use braid_stats::MixtureType;
use once_cell::sync::OnceCell;
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::{
    Categorical, Gamma, Gaussian, Mixture, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{impl_metadata_version, MetadataVersion};

pub const METADATA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub states: Vec<DatalessState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_ids: Option<Vec<usize>>,
    pub codebook: Codebook,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataStore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng: Option<Xoshiro256Plus>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessState {
    pub views: Vec<DatalessView>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub view_alpha_prior: CrpPrior,
    pub loglike: f64,
    #[serde(default)]
    pub log_prior: f64,
    #[serde(default)]
    pub log_view_alpha_prior: f64,
    #[serde(default)]
    pub log_state_alpha_prior: f64,
    pub diagnostics: StateDiagnostics,
}

/// Marks a state as having no data in its columns
pub struct EmptyState(pub State);

impl From<braid_cc::state::State> for DatalessState {
    fn from(mut state: braid_cc::state::State) -> DatalessState {
        DatalessState {
            views: state.views.drain(..).map(|view| view.into()).collect(),
            asgn: state.asgn,
            weights: state.weights,
            view_alpha_prior: state.view_alpha_prior,
            loglike: state.loglike,
            log_prior: state.log_prior,
            log_view_alpha_prior: state.log_view_alpha_prior,
            log_state_alpha_prior: state.log_state_alpha_prior,
            diagnostics: state.diagnostics.into(),
        }
    }
}

impl From<DatalessState> for EmptyState {
    fn from(mut dl_state: DatalessState) -> EmptyState {
        let views = dl_state
            .views
            .drain(..)
            .map(|mut dl_view| {
                let mut ftr_ids: Vec<usize> =
                    dl_view.ftrs.keys().copied().collect();

                let ftrs: BTreeMap<usize, ColModel> = ftr_ids
                    .drain(..)
                    .map(|id| {
                        let dl_ftr = dl_view.ftrs.remove(&id).unwrap();
                        let cm: ColModel = match dl_ftr {
                            DatalessColModel::Continuous(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Continuous(ecm.0)
                            }
                            DatalessColModel::Categorical(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Categorical(ecm.0)
                            }
                            DatalessColModel::Labeler(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Labeler(ecm.0)
                            }
                            DatalessColModel::Count(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Count(ecm.0)
                            }
                        };
                        (id, cm)
                    })
                    .collect();

                View {
                    asgn: dl_view.asgn.into(),
                    weights: dl_view.weights,
                    ftrs,
                }
            })
            .collect();

        EmptyState(State {
            views,
            asgn: dl_state.asgn.into(),
            weights: dl_state.weights,
            view_alpha_prior: dl_state.view_alpha_prior,
            loglike: dl_state.loglike,
            log_prior: dl_state.log_prior,
            log_view_alpha_prior: dl_state.log_view_alpha_prior,
            log_state_alpha_prior: dl_state.log_state_alpha_prior,
            diagnostics: dl_state.diagnostics.into(),
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessView {
    pub ftrs: BTreeMap<usize, DatalessColModel>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
}

impl From<View> for DatalessView {
    fn from(mut view: View) -> DatalessView {
        DatalessView {
            ftrs: {
                let keys: Vec<usize> = view.ftrs.keys().cloned().collect();
                keys.iter()
                    .map(|k| (*k, view.ftrs.remove(k).unwrap().into()))
                    .collect()
            },
            asgn: view.asgn,
            weights: view.weights,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub enum DatalessColModel {
    Continuous(DatalessColumn<f64, Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(DatalessColumn<u8, Categorical, SymmetricDirichlet, CsdHyper>),
    Labeler(DatalessColumn<Label, Labeler, LabelerPrior, ()>),
    Count(DatalessColumn<u32, Poisson, Gamma, PgHyper>),
}

impl From<ColModel> for DatalessColModel {
    fn from(col_model: ColModel) -> DatalessColModel {
        match col_model {
            ColModel::Categorical(col) => {
                DatalessColModel::Categorical(col.into())
            }
            ColModel::Continuous(col) => {
                DatalessColModel::Continuous(col.into())
            }
            ColModel::Labeler(col) => DatalessColModel::Labeler(col.into()),
            ColModel::Count(col) => DatalessColModel::Count(col.into()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessColumn<X, Fx, Pr, H>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    MixtureType: From<Mixture<Fx>>,
    Fx::Stat: BraidStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    id: usize,
    #[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
    components: Vec<ConjugateComponent<X, Fx, Pr>>,
    #[serde(bound(deserialize = "Pr: serde::de::DeserializeOwned"))]
    prior: Pr,
    #[serde(bound(deserialize = "H: serde::de::DeserializeOwned"))]
    hyper: H,
    #[serde(default)]
    ignore_hyper: bool,
}

macro_rules! col2dataless {
    ($x:ty, $fx:ty, $pr:ty, $h:ty) => {
        impl From<Column<$x, $fx, $pr, $h>>
            for DatalessColumn<$x, $fx, $pr, $h>
        {
            fn from(col: Column<$x, $fx, $pr, $h>) -> Self {
                DatalessColumn {
                    id: col.id,
                    components: col.components,
                    prior: col.prior,
                    hyper: col.hyper,
                    ignore_hyper: col.ignore_hyper,
                }
            }
        }
    };
}

col2dataless!(f64, Gaussian, NormalInvChiSquared, NixHyper);
col2dataless!(u8, Categorical, SymmetricDirichlet, CsdHyper);
col2dataless!(Label, Labeler, LabelerPrior, ());
col2dataless!(u32, Poisson, Gamma, PgHyper);

struct EmptyColumn<X, Fx, Pr, H>(Column<X, Fx, Pr, H>)
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: BraidPrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
    MixtureType: From<Mixture<Fx>>;

macro_rules! dataless2col {
    ($x:ty, $fx:ty, $pr:ty, $h:ty) => {
        impl Into<EmptyColumn<$x, $fx, $pr, $h>>
            for DatalessColumn<$x, $fx, $pr, $h>
        {
            fn into(self) -> EmptyColumn<$x, $fx, $pr, $h> {
                EmptyColumn(Column {
                    id: self.id,
                    components: self.components,
                    prior: self.prior,
                    hyper: self.hyper,
                    data: SparseContainer::default(),
                    ln_m_cache: OnceCell::new(),
                    ignore_hyper: self.ignore_hyper,
                })
            }
        }
    };
}

dataless2col!(f64, Gaussian, NormalInvChiSquared, NixHyper);
dataless2col!(u8, Categorical, SymmetricDirichlet, CsdHyper);
dataless2col!(Label, Labeler, LabelerPrior, ());
dataless2col!(u32, Poisson, Gamma, PgHyper);

impl<X, Fx, Pr, H> MetadataVersion for DatalessColumn<X, Fx, Pr, H>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    MixtureType: From<Mixture<Fx>>,
    Fx::Stat: BraidStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn metadata_version() -> u32 {
        METADATA_VERSION
    }
}

impl_metadata_version!(DatalessColModel, METADATA_VERSION);
impl_metadata_version!(DatalessView, METADATA_VERSION);
impl_metadata_version!(DatalessState, METADATA_VERSION);
impl_metadata_version!(Metadata, METADATA_VERSION);

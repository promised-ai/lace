use std::collections::BTreeMap;

use lace_cc::component::ConjugateComponent;
use lace_cc::feature::{ColModel, Column};
use lace_cc::state::StateDiagnostics;
use lace_cc::traits::{LaceDatum, LaceLikelihood, LacePrior, LaceStat};
use lace_data::{FeatureData, SparseContainer};
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::rv::dist::{
    Bernoulli, Beta, Categorical, Gamma, Gaussian, Mixture,
    NormalInvChiSquared, Poisson, SymmetricDirichlet,
};
use lace_stats::MixtureType;

use rand_xoshiro::Xoshiro256Plus;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::OnceLock;

use crate::{impl_metadata_version, MetadataVersion};

pub const METADATA_VERSION: i32 = 0;

// #[derive(Serialize, Deserialize, Debug)]
// #[serde(deny_unknown_fields)]
// pub struct Codebook(pub lace_codebook::Codebook);

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub alpha: f64,
    pub asgn: Vec<usize>,
    pub counts: Vec<usize>,
    pub n_cats: usize,
    pub prior: Gamma,
}

/// Codebook object for storing information about the dataset
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Codebook {
    pub table_name: String,
    pub state_alpha_prior: Option<Gamma>,
    pub view_alpha_prior: Option<Gamma>,
    pub col_metadata: lace_codebook::ColMetadataList,
    pub comments: Option<String>,
    pub row_names: lace_codebook::RowNameList,
}

// to_from_newtype!(lace_codebook::Codebook, Codebook);

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub states: Vec<DatalessStateAndDiagnostics>,
    pub state_ids: Vec<usize>,
    pub codebook: Codebook,
    pub data: Option<DataStore>,
    pub rng: Option<Xoshiro256Plus>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DataStore(BTreeMap<usize, FeatureData>);

impl From<lace_data::DataStore> for DataStore {
    fn from(data: lace_data::DataStore) -> Self {
        Self(data.0)
    }
}

impl From<DataStore> for lace_data::DataStore {
    fn from(data: DataStore) -> Self {
        Self(data.0)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DatalessStateAndDiagnostics {
    pub state: DatalessState,
    pub diagnostics: StateDiagnostics,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessState {
    pub views: Vec<DatalessView>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub view_alpha_prior: Gamma,
    pub loglike: f64,
    #[serde(default)]
    pub log_prior: f64,
    #[serde(default)]
    pub log_view_alpha_prior: f64,
    #[serde(default)]
    pub log_state_alpha_prior: f64,
}

// /// Marks a state as having no data in its columns
// pub struct EmptyState(pub State);

// impl From<lace_cc::state::State> for DatalessStateAndDiagnostics {
//     fn from(mut state: lace_cc::state::State) -> Self {
//         Self {
//             state: DatalessState {
//                 views: state.views.drain(..).map(|view| view.into()).collect(),
//                 asgn: state.asgn,
//                 weights: state.weights,
//                 view_alpha_prior: state.view_alpha_prior,
//                 loglike: state.loglike,
//                 log_prior: state.log_prior,
//                 log_view_alpha_prior: state.log_view_alpha_prior,
//                 log_state_alpha_prior: state.log_state_alpha_prior,
//             },
//             diagnostics: state.diagnostics,
//         }
//     }
// }

// impl From<DatalessStateAndDiagnostics> for EmptyState {
//     fn from(mut dl_state: DatalessStateAndDiagnostics) -> EmptyState {
//         let views = dl_state
//             .state
//             .views
//             .drain(..)
//             .map(|mut dl_view| {
//                 let mut ftr_ids: Vec<usize> =
//                     dl_view.ftrs.keys().copied().collect();

//                 let ftrs: BTreeMap<usize, ColModel> = ftr_ids
//                     .drain(..)
//                     .map(|id| {
//                         let dl_ftr = dl_view.ftrs.remove(&id).unwrap();
//                         let cm: ColModel = match dl_ftr {
//                             DatalessColModel::Continuous(cm) => {
//                                 let ecm: EmptyColumn<_, _, _, _> = cm.into();
//                                 ColModel::Continuous(ecm.0)
//                             }
//                             DatalessColModel::Categorical(cm) => {
//                                 let ecm: EmptyColumn<_, _, _, _> = cm.into();
//                                 ColModel::Categorical(ecm.0)
//                             }
//                             DatalessColModel::Count(cm) => {
//                                 let ecm: EmptyColumn<_, _, _, _> = cm.into();
//                                 ColModel::Count(ecm.0)
//                             }
//                             DatalessColModel::MissingNotAtRandom(mnar) => {
//                                 let fx: ColModel = (*mnar.fx).into();
//                                 let missing: EmptyColumn<_, _, _, _> =
//                                     mnar.missing.into();
//                                 ColModel::MissingNotAtRandom(
//                                     MissingNotAtRandom {
//                                         fx: Box::new(fx),
//                                         present: missing.0,
//                                     },
//                                 )
//                             }
//                         };
//                         (id, cm)
//                     })
//                     .collect();

//                 View {
//                     asgn: dl_view.asgn,
//                     weights: dl_view.weights,
//                     ftrs,
//                 }
//             })
//             .collect();

//         EmptyState(State {
//             views,
//             asgn: dl_state.state.asgn,
//             weights: dl_state.state.weights,
//             view_alpha_prior: dl_state.state.view_alpha_prior,
//             loglike: dl_state.state.loglike,
//             log_prior: dl_state.state.log_prior,
//             log_view_alpha_prior: dl_state.state.log_view_alpha_prior,
//             log_state_alpha_prior: dl_state.state.log_state_alpha_prior,
//             diagnostics: dl_state.diagnostics,
//         })
//     }
// }

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessView {
    pub ftrs: BTreeMap<usize, DatalessColModel>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
}

// impl From<View> for DatalessView {
//     fn from(mut view: View) -> DatalessView {
//         DatalessView {
//             ftrs: {
//                 let keys: Vec<usize> = view.ftrs.keys().cloned().collect();
//                 keys.iter()
//                     .map(|k| (*k, view.ftrs.remove(k).unwrap().into()))
//                     .collect()
//             },
//             asgn: view.asgn,
//             weights: view.weights,
//         }
//     }
// }

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub enum DatalessColModel {
    Continuous(DatalessColumn<f64, Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(DatalessColumn<u8, Categorical, SymmetricDirichlet, CsdHyper>),
    Count(DatalessColumn<u32, Poisson, Gamma, PgHyper>),
    MissingNotAtRandom(DatalessMissingNotAtRandom),
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
            ColModel::Count(col) => DatalessColModel::Count(col.into()),
            ColModel::MissingNotAtRandom(mnar) => {
                DatalessColModel::MissingNotAtRandom(
                    DatalessMissingNotAtRandom {
                        fx: Box::new((*mnar.fx).into()),
                        missing: mnar.present.into(),
                    },
                )
            }
        }
    }
}

impl From<DatalessColModel> for ColModel {
    fn from(col_model: DatalessColModel) -> Self {
        match col_model {
            DatalessColModel::Continuous(cm) => {
                let empty_col: EmptyColumn<_, _, _, _> = cm.into();
                Self::Continuous(empty_col.0)
            }
            DatalessColModel::Count(cm) => {
                let empty_col: EmptyColumn<_, _, _, _> = cm.into();
                Self::Count(empty_col.0)
            }
            DatalessColModel::Categorical(cm) => {
                let empty_col: EmptyColumn<_, _, _, _> = cm.into();
                Self::Categorical(empty_col.0)
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessMissingNotAtRandom {
    pub fx: Box<DatalessColModel>,
    pub missing: DatalessColumn<bool, Bernoulli, Beta, ()>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessColumn<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    MixtureType: From<Mixture<Fx>>,
    Fx::Stat: LaceStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    pub id: usize,
    #[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
    pub components: Vec<ConjugateComponent<X, Fx, Pr>>,
    #[serde(bound(deserialize = "Pr: serde::de::DeserializeOwned"))]
    pub prior: Pr,
    #[serde(bound(deserialize = "H: serde::de::DeserializeOwned"))]
    pub hyper: H,
    #[serde(default)]
    pub ignore_hyper: bool,
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
col2dataless!(u32, Poisson, Gamma, PgHyper);
col2dataless!(bool, Bernoulli, Beta, ());

pub struct EmptyColumn<X, Fx, Pr, H>(pub Column<X, Fx, Pr, H>)
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
    MixtureType: From<Mixture<Fx>>;

macro_rules! dataless2col {
    ($x:ty, $fx:ty, $pr:ty, $h:ty) => {
        impl From<DatalessColumn<$x, $fx, $pr, $h>>
            for EmptyColumn<$x, $fx, $pr, $h>
        {
            fn from(
                col_dl: DatalessColumn<$x, $fx, $pr, $h>,
            ) -> EmptyColumn<$x, $fx, $pr, $h> {
                EmptyColumn(Column {
                    id: col_dl.id,
                    components: col_dl.components,
                    prior: col_dl.prior,
                    hyper: col_dl.hyper,
                    data: SparseContainer::default(),
                    ln_m_cache: OnceLock::new(),
                    ignore_hyper: col_dl.ignore_hyper,
                })
            }
        }
    };
}

dataless2col!(f64, Gaussian, NormalInvChiSquared, NixHyper);
dataless2col!(u8, Categorical, SymmetricDirichlet, CsdHyper);
dataless2col!(u32, Poisson, Gamma, PgHyper);
dataless2col!(bool, Bernoulli, Beta, ());

impl_metadata_version!(Metadata, METADATA_VERSION);
impl_metadata_version!(Codebook, METADATA_VERSION);
impl_metadata_version!(DatalessColModel, METADATA_VERSION);
impl_metadata_version!(DatalessView, METADATA_VERSION);
impl_metadata_version!(DatalessState, METADATA_VERSION);
impl_metadata_version!(DataStore, METADATA_VERSION);

// Create the loaders module for latest
crate::loaders!(
    DatalessStateAndDiagnostics,
    DataStore,
    Codebook,
    rand_xoshiro::Xoshiro256Plus
);

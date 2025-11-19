use std::collections::BTreeMap;
use std::sync::OnceLock;

use rand_xoshiro::Xoshiro256Plus;
use rv::dist::Bernoulli;
use rv::dist::Beta;
use rv::dist::Categorical;
use rv::dist::Gamma;
use rv::dist::Gaussian;
use rv::dist::Mixture;
use rv::dist::NormalInvChiSquared;
use rv::dist::Poisson;
use rv::dist::SymmetricDirichlet;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Serialize;

use crate::cc::component::ConjugateComponent;
use crate::cc::feature::ColModel;
use crate::cc::feature::Column;
use crate::cc::feature::MissingNotAtRandom;
use crate::cc::state::State;
use crate::cc::state::StateDiagnostics;
use crate::cc::state::StateScoreComponents;
use crate::cc::traits::LaceDatum;
use crate::cc::traits::LaceLikelihood;
use crate::cc::traits::LacePrior;
use crate::cc::traits::LaceStat;
use crate::cc::view::View;
use crate::data::FeatureData;
use crate::data::SparseContainer;
use crate::impl_metadata_version;
use crate::metadata::MetadataVersion;
use crate::stats::prior::csd::CsdHyper;
use crate::stats::prior::nix::NixHyper;
use crate::stats::prior::pg::PgHyper;
use crate::stats::prior_process::PriorProcess;
use crate::stats::MixtureType;
use crate::to_from_newtype;

pub const METADATA_VERSION: i32 = 2;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Codebook(pub crate::codebook::Codebook);

to_from_newtype!(crate::codebook::Codebook, Codebook);

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

impl From<crate::data::DataStore> for DataStore {
    fn from(data: crate::data::DataStore) -> Self {
        Self(data.0)
    }
}

impl From<DataStore> for crate::data::DataStore {
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
    pub prior_process: PriorProcess,
    pub weights: Vec<f64>,
    pub score: StateScoreComponents,
}

/// Marks a state as having no data in its columns
pub struct EmptyState(pub State);

impl From<crate::cc::state::State> for DatalessStateAndDiagnostics {
    fn from(mut state: crate::cc::state::State) -> Self {
        Self {
            state: DatalessState {
                views: state.views.drain(..).map(|view| view.into()).collect(),
                prior_process: state.prior_process,
                weights: state.weights,
                score: state.score,
            },
            diagnostics: state.diagnostics,
        }
    }
}

impl From<DatalessStateAndDiagnostics> for EmptyState {
    fn from(mut dl_state: DatalessStateAndDiagnostics) -> EmptyState {
        let views = dl_state
            .state
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
                            DatalessColModel::Count(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Count(ecm.0)
                            }
                            DatalessColModel::MissingNotAtRandom(mnar) => {
                                let fx: ColModel = (*mnar.fx).into();
                                let missing: EmptyColumn<_, _, _, _> =
                                    mnar.missing.into();
                                ColModel::MissingNotAtRandom(
                                    MissingNotAtRandom {
                                        fx: Box::new(fx),
                                        present: missing.0,
                                    },
                                )
                            }
                        };
                        (id, cm)
                    })
                    .collect();

                View {
                    prior_process: dl_view.prior_process,
                    weights: dl_view.weights,
                    ftrs,
                }
            })
            .collect();

        EmptyState(State {
            views,
            prior_process: dl_state.state.prior_process,
            weights: dl_state.state.weights,
            score: dl_state.state.score,
            diagnostics: dl_state.diagnostics,
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessView {
    pub ftrs: BTreeMap<usize, DatalessColModel>,
    pub prior_process: PriorProcess,
    pub weights: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub enum DatalessColModel {
    Continuous(DatalessColumn<f64, Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(DatalessColumn<u32, Categorical, SymmetricDirichlet, CsdHyper>),
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
    Pr::MCache: Clone + std::fmt::Debug,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
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

pub struct EmptyColumn<X, Fx, Pr, H>(pub Column<X, Fx, Pr, H>)
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    Pr::MCache: Clone + std::fmt::Debug,
    Pr::PpCache: Send + Sync + Clone + std::fmt::Debug,
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
dataless2col!(u32, Categorical, SymmetricDirichlet, CsdHyper);
dataless2col!(u32, Poisson, Gamma, PgHyper);
dataless2col!(bool, Bernoulli, Beta, ());

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
col2dataless!(u32, Categorical, SymmetricDirichlet, CsdHyper);
col2dataless!(u32, Poisson, Gamma, PgHyper);
col2dataless!(bool, Bernoulli, Beta, ());

impl From<View> for DatalessView {
    fn from(mut view: View) -> DatalessView {
        DatalessView {
            ftrs: {
                let keys: Vec<usize> = view.ftrs.keys().cloned().collect();
                keys.iter()
                    .map(|k| (*k, view.ftrs.remove(k).unwrap().into()))
                    .collect()
            },
            prior_process: view.prior_process,
            weights: view.weights,
        }
    }
}

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

use std::collections::BTreeMap;

use lace_cc::component::ConjugateComponent;
use lace_cc::feature::{ColModel, Column, MissingNotAtRandom};
use lace_cc::prior_process::PriorProcess;
use lace_cc::state::{State, StateDiagnostics, StateScoreComponents};
use lace_cc::traits::{LaceDatum, LaceLikelihood, LacePrior, LaceStat};
use lace_cc::view::View;
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

use crate::versions::v1;
use crate::{impl_metadata_version, to_from_newtype, MetadataVersion};

pub const METADATA_VERSION: i32 = 1;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Codebook(pub lace_codebook::Codebook);

to_from_newtype!(lace_codebook::Codebook, Codebook);

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub states: Vec<DatalessStateAndDiagnostics>,
    pub state_ids: Vec<usize>,
    pub codebook: Codebook,
    pub data: Option<v1::DataStore>,
    pub rng: Option<Xoshiro256Plus>,
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

impl From<lace_cc::state::State> for DatalessStateAndDiagnostics {
    fn from(mut state: lace_cc::state::State) -> Self {
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
                            v1::DatalessColModel::Continuous(cm) => {
                                let ecm: v1::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Continuous(ecm.0)
                            }
                            v1::DatalessColModel::Categorical(cm) => {
                                let ecm: v1::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Categorical(ecm.0)
                            }
                            v1::DatalessColModel::Count(cm) => {
                                let ecm: v1::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Count(ecm.0)
                            }
                            v1::DatalessColModel::MissingNotAtRandom(mnar) => {
                                let fx: ColModel = (*mnar.fx).into();
                                let missing: v1::EmptyColumn<_, _, _, _> =
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
    pub ftrs: BTreeMap<usize, v1::DatalessColModel>,
    pub prior_process: PriorProcess,
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
            prior_process: view.prior_process,
            weights: view.weights,
        }
    }
}

impl From<v1::DatalessStateAndDiagnostics> for DatalessStateAndDiagnostics {
    fn from(state_and_diag: v1::DatalessStateAndDiagnostics) -> Self {
        Self {
            state: state_and_diag.state.into(),
            diagnostics: state_and_diag.diagnostics.into(),
        }
    }
}

impl From<v1::Codebook> for Codebook {
    fn from(codebook: v1::Codebook) -> Self {
        Self(lace_codebook::Codebook { 
            table_name: codebook.0.table_name, 
            state_alpha_prior: codebook,
            view_alpha_prior: ,
            col_metadata: ,
            comments: ,
            row_names:  
        })
    }
}

impl From<v1::Metadata> for Metadata {
    fn from(mut metadata: v1::Metadata) -> Self {
        Self {
            states: metadata
                .states
                .drain(..)
                .map(|state| state.into())
                .collect(),
            state_ids: metadata.state_ids,
            codebook: metadata.codebook.into(),
            data: metadata.codebook,
            rng: metadata.rng,
        }
    }
}

impl_metadata_version!(Metadata, METADATA_VERSION);
impl_metadata_version!(Codebook, METADATA_VERSION);
impl_metadata_version!(DatalessView, METADATA_VERSION);
impl_metadata_version!(DatalessState, METADATA_VERSION);

// Create the loaders module for latest
crate::loaders!(
    DatalessStateAndDiagnostics,
    v1::DataStore,
    Codebook,
    rand_xoshiro::Xoshiro256Plus
);

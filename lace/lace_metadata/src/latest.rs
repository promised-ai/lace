use std::collections::BTreeMap;

use lace_cc::assignment::Assignment;
use lace_cc::feature::{ColModel, Latent, MissingNotAtRandom};
use lace_cc::state::{State, StateDiagnostics};
use lace_cc::view::View;
use lace_codebook::ColMetadata;
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};

#[cfg(feature = "experimental")]
use lace_cc::feature::Column;
#[cfg(feature = "experimental")]
use lace_data::SparseContainer;
#[cfg(feature = "experimental")]
use lace_stats::experimental::sbd::SbdHyper;
#[cfg(feature = "experimental")]
use lace_stats::rv::experimental::{Sb, Sbd};
#[cfg(feature = "experimental")]
use once_cell::sync::OnceCell;

use crate::versions::v0;
use crate::versions::v0::{DataStore, DatalessMissingNotAtRandom};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};

use crate::{impl_metadata_version, to_from_newtype, MetadataVersion};

pub const METADATA_VERSION: i32 = 1;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Codebook(pub lace_codebook::Codebook);

to_from_newtype!(lace_codebook::Codebook, Codebook);

#[derive(Serialize, Deserialize, Debug)]
pub struct DatalessStateAndDiagnostics {
    pub state: DatalessState,
    pub diagnostics: StateDiagnostics,
}

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

/// Marks a state as having no data in its columns
pub struct EmptyState(pub State);

impl From<lace_cc::state::State> for DatalessStateAndDiagnostics {
    fn from(mut state: lace_cc::state::State) -> Self {
        Self {
            state: DatalessState {
                views: state.views.drain(..).map(|view| view.into()).collect(),
                asgn: state.asgn,
                weights: state.weights,
                view_alpha_prior: state.view_alpha_prior,
                loglike: state.loglike,
                log_prior: state.log_prior,
                log_view_alpha_prior: state.log_view_alpha_prior,
                log_state_alpha_prior: state.log_state_alpha_prior,
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
                                let ecm: v0::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Continuous(ecm.0)
                            }
                            DatalessColModel::Categorical(cm) => {
                                let ecm: v0::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Categorical(ecm.0)
                            }
                            DatalessColModel::Count(cm) => {
                                let ecm: v0::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Count(ecm.0)
                            }
                            DatalessColModel::MissingNotAtRandom(mnar) => {
                                let fx: ColModel = (*mnar.fx).into();
                                let missing: v0::EmptyColumn<_, _, _, _> =
                                    mnar.missing.into();
                                ColModel::MissingNotAtRandom(
                                    MissingNotAtRandom {
                                        fx: Box::new(fx),
                                        present: missing.0,
                                    },
                                )
                            }
                            DatalessColModel::Latent(latent) => {
                                let column: ColModel = (*latent.column).into();
                                ColModel::Latent(Latent {
                                    column: Box::new(column),
                                    assignment: dl_view.asgn.asgn.clone(),
                                })
                            }
                            #[cfg(feature = "experimental")]
                            DatalessColModel::Index(cm) => {
                                let ecm: v0::EmptyColumn<_, _, _, _> =
                                    cm.into();
                                ColModel::Index(ecm.0)
                            }
                        };
                        (id, cm)
                    })
                    .collect();

                View {
                    asgn: dl_view.asgn,
                    weights: dl_view.weights,
                    ftrs,
                }
            })
            .collect();

        EmptyState(State {
            views,
            asgn: dl_state.state.asgn,
            weights: dl_state.state.weights,
            view_alpha_prior: dl_state.state.view_alpha_prior,
            loglike: dl_state.state.loglike,
            log_prior: dl_state.state.log_prior,
            log_view_alpha_prior: dl_state.state.log_view_alpha_prior,
            log_state_alpha_prior: dl_state.state.log_state_alpha_prior,
            diagnostics: dl_state.diagnostics,
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
    Continuous(
        v0::DatalessColumn<f64, Gaussian, NormalInvChiSquared, NixHyper>,
    ),
    Categorical(
        v0::DatalessColumn<u8, Categorical, SymmetricDirichlet, CsdHyper>,
    ),
    Count(v0::DatalessColumn<u32, Poisson, Gamma, PgHyper>),
    MissingNotAtRandom(v0::DatalessMissingNotAtRandom),
    Latent(DatalessLatent),
    #[cfg(feature = "experimental")]
    Index(v0::DatalessColumn<usize, Sbd, Sb, SbdHyper>),
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
            ColModel::Latent(latent) => {
                DatalessColModel::Latent(DatalessLatent {
                    column: Box::new((*latent.column).into()),
                })
            }
            #[cfg(feature = "experimental")]
            ColModel::Index(col) => DatalessColModel::Index(col.into()),
        }
    }
}

impl From<DatalessColModel> for ColModel {
    fn from(col_model: DatalessColModel) -> Self {
        match col_model {
            DatalessColModel::Continuous(cm) => {
                let empty_col: v0::EmptyColumn<_, _, _, _> = cm.into();
                Self::Continuous(empty_col.0)
            }
            DatalessColModel::Count(cm) => {
                let empty_col: v0::EmptyColumn<_, _, _, _> = cm.into();
                Self::Count(empty_col.0)
            }
            DatalessColModel::Categorical(cm) => {
                let empty_col: v0::EmptyColumn<_, _, _, _> = cm.into();
                Self::Categorical(empty_col.0)
            }
            #[cfg(feature = "experimental")]
            DatalessColModel::Index(cm) => {
                let empty_col: v0::EmptyColumn<_, _, _, _> = cm.into();
                Self::Index(empty_col.0)
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessLatent {
    column: Box<DatalessColModel>,
}

#[cfg(feature = "experimental")]
crate::col2dataless!(usize, Sbd, Sb, SbdHyper);
#[cfg(feature = "experimental")]
crate::dataless2col!(usize, Sbd, Sb, SbdHyper);

impl From<v0::DatalessColModel> for DatalessColModel {
    fn from(col: v0::DatalessColModel) -> Self {
        match col {
            v0::DatalessColModel::Continuous(inner) => Self::Continuous(inner),
            v0::DatalessColModel::Categorical(inner) => {
                Self::Categorical(inner)
            }
            v0::DatalessColModel::Count(inner) => Self::Count(inner),
            v0::DatalessColModel::MissingNotAtRandom(inner) => {
                Self::MissingNotAtRandom(inner)
            }
        }
    }
}

impl From<v0::DatalessView> for DatalessView {
    fn from(mut view: v0::DatalessView) -> Self {
        let mut ftrs = BTreeMap::new();
        while let Some((ix, ftr)) = view.ftrs.pop_last() {
            let cm: DatalessColModel = ftr.into();
            ftrs.insert(ix, cm);
        }
        DatalessView {
            ftrs,
            asgn: view.asgn,
            weights: view.weights,
        }
    }
}

impl From<v0::DatalessState> for DatalessState {
    fn from(mut state: v0::DatalessState) -> Self {
        DatalessState {
            views: state.views.drain(..).map(DatalessView::from).collect(),
            asgn: state.asgn,
            weights: state.weights,
            view_alpha_prior: state.view_alpha_prior,
            loglike: state.loglike,
            log_prior: state.log_prior,
            log_view_alpha_prior: state.log_view_alpha_prior,
            log_state_alpha_prior: state.log_state_alpha_prior,
        }
    }
}

impl From<v0::DatalessStateAndDiagnostics> for DatalessStateAndDiagnostics {
    fn from(state_and_diag: v0::DatalessStateAndDiagnostics) -> Self {
        Self {
            state: state_and_diag.state.into(),
            diagnostics: state_and_diag.diagnostics,
        }
    }
}

impl From<v0::ColMetadata> for ColMetadata {
    fn from(colmd: v0::ColMetadata) -> Self {
        ColMetadata {
            name: colmd.name,
            coltype: colmd.coltype,
            notes: colmd.notes,
            missing_not_at_random: colmd.missing_not_at_random,
            latent: false,
        }
    }
}

impl From<v0::Codebook> for Codebook {
    fn from(mut codebook: v0::Codebook) -> Self {
        Self(lace_codebook::Codebook {
            table_name: codebook.table_name,
            state_alpha_prior: codebook.state_alpha_prior,
            view_alpha_prior: codebook.view_alpha_prior,
            col_metadata: codebook
                .col_metadata
                .drain(..)
                .map(ColMetadata::from)
                .collect::<Vec<ColMetadata>>()
                .try_into()
                .expect("Vec<v0::ColMetadata> was invalid"), //
            comments: codebook.comments,
            row_names: codebook.row_names,
        })
    }
}

impl From<v0::Metadata> for Metadata {
    fn from(mut metadata: v0::Metadata) -> Self {
        Metadata {
            states: metadata
                .states
                .drain(..)
                .map(DatalessStateAndDiagnostics::from)
                .collect(),
            state_ids: metadata.state_ids,
            codebook: metadata.codebook.into(),
            data: metadata.data,
            rng: metadata.rng,
        }
    }
}

impl_metadata_version!(Metadata, METADATA_VERSION);
impl_metadata_version!(Codebook, METADATA_VERSION);
impl_metadata_version!(DatalessColModel, METADATA_VERSION);
impl_metadata_version!(DatalessView, METADATA_VERSION);
impl_metadata_version!(DatalessState, METADATA_VERSION);

// Create the loaders module for latest
crate::loaders!(
    DatalessStateAndDiagnostics,
    DataStore,
    Codebook,
    rand_xoshiro::Xoshiro256Plus
);

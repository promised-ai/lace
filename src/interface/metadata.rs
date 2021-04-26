use crate::cc::state::StateDiagnostics;
use crate::cc::{
    Assignment, ColModel, Column, ConjugateComponent, DataStore, State, View,
};
use crate::dist::{BraidDatum, BraidLikelihood, BraidPrior, BraidStat};
use crate::{DatalessOracle, Engine, Oracle};
use braid_codebook::Codebook;
use braid_data::SparseContainer;
use braid_stats::labeler::{Label, Labeler, LabelerPrior};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use braid_stats::MixtureType;
use once_cell::sync::OnceCell;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::{
    Categorical, Gamma, Gaussian, Mixture, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use rv::traits::ConjugatePrior;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::BTreeMap;
use std::convert::TryFrom;
use thiserror::Error;

/// Braid metadata. Intermediate struct for serializing and deserializing
/// Engines and Oracles.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    states: Vec<DatalessState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state_ids: Option<Vec<usize>>,
    codebook: Codebook,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<DataStore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rng: Option<Xoshiro256Plus>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct DatalessState {
    views: Vec<DatalessView>,
    asgn: Assignment,
    weights: Vec<f64>,
    view_alpha_prior: CrpPrior,
    loglike: f64,
    #[serde(default)]
    log_prior: f64,
    #[serde(default)]
    log_view_alpha_prior: f64,
    #[serde(default)]
    log_state_alpha_prior: f64,
    diagnostics: StateDiagnostics,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct DatalessView {
    ftrs: BTreeMap<usize, DatalessColModel>,
    asgn: Assignment,
    weights: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
enum DatalessColModel {
    Continuous(DatalessColumn<f64, Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(DatalessColumn<u8, Categorical, SymmetricDirichlet, CsdHyper>),
    Labeler(DatalessColumn<Label, Labeler, LabelerPrior, ()>),
    Count(DatalessColumn<u32, Poisson, Gamma, PgHyper>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
#[serde(deny_unknown_fields)]
struct DatalessColumn<X, Fx, Pr, H>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: BraidPrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    id: usize,
    components: Vec<ConjugateComponent<X, Fx, Pr>>,
    prior: Pr,
    hyper: H,
    #[serde(default)]
    ignore_hyper: bool,
    #[serde(skip)]
    ln_m_cache: OnceCell<<Pr as ConjugatePrior<X, Fx>>::LnMCache>,
}

impl Into<DatalessState> for State {
    fn into(mut self) -> DatalessState {
        DatalessState {
            views: self.views.drain(..).map(|view| view.into()).collect(),
            asgn: self.asgn,
            weights: self.weights,
            view_alpha_prior: self.view_alpha_prior,
            loglike: self.loglike,
            log_prior: self.log_prior,
            log_view_alpha_prior: self.log_view_alpha_prior,
            log_state_alpha_prior: self.log_state_alpha_prior,
            diagnostics: self.diagnostics,
        }
    }
}

impl Into<DatalessView> for View {
    fn into(mut self) -> DatalessView {
        DatalessView {
            ftrs: {
                let keys: Vec<usize> = self.ftrs.keys().cloned().collect();
                keys.iter()
                    .map(|k| (*k, self.ftrs.remove(k).unwrap().into()))
                    .collect()
            },
            asgn: self.asgn,
            weights: self.weights,
        }
    }
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
                    ln_m_cache: OnceCell::new(),
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

impl Into<DatalessColModel> for ColModel {
    fn into(self) -> DatalessColModel {
        match self {
            ColModel::Categorical(cm) => {
                DatalessColModel::Categorical(cm.into())
            }
            ColModel::Continuous(cm) => DatalessColModel::Continuous(cm.into()),
            ColModel::Labeler(cm) => DatalessColModel::Labeler(cm.into()),
            ColModel::Count(cm) => DatalessColModel::Count(cm.into()),
        }
    }
}

impl From<Engine> for Metadata {
    fn from(mut engine: Engine) -> Metadata {
        let data = Some(DataStore(engine.states[0].take_data()));
        Metadata {
            states: engine.states.drain(..).map(|state| state.into()).collect(),
            state_ids: Some(engine.state_ids),
            codebook: engine.codebook,
            rng: Some(engine.rng),
            data,
        }
    }
}

struct EmptyState(State);

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
                    asgn: dl_view.asgn,
                    weights: dl_view.weights,
                    ftrs,
                }
            })
            .collect();

        EmptyState(State {
            views,
            asgn: dl_state.asgn,
            weights: dl_state.weights,
            view_alpha_prior: dl_state.view_alpha_prior,
            loglike: dl_state.loglike,
            log_prior: dl_state.log_prior,
            log_view_alpha_prior: dl_state.log_view_alpha_prior,
            log_state_alpha_prior: dl_state.log_state_alpha_prior,
            diagnostics: dl_state.diagnostics,
        })
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("Cannot deserialize with data field `None`")]
pub struct DataFieldNoneError;

impl From<Oracle> for Metadata {
    fn from(mut oracle: Oracle) -> Metadata {
        Metadata {
            states: oracle.states.drain(..).map(|state| state.into()).collect(),
            state_ids: None,
            codebook: oracle.codebook,
            data: Some(oracle.data),
            rng: None,
        }
    }
}

impl TryFrom<Metadata> for Engine {
    type Error = DataFieldNoneError;
    fn try_from(mut md: Metadata) -> Result<Engine, Self::Error> {
        let data = md.data.ok_or_else(|| DataFieldNoneError)?.0;

        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: EmptyState = dl_state.into();
                let mut state = empty_state.0;
                state.repop_data(data.clone());
                state
            })
            .collect();

        let state_ids = md
            .state_ids
            .unwrap_or_else(|| (0..states.len()).collect::<Vec<usize>>());
        let rng = md.rng.unwrap_or_else(Xoshiro256Plus::from_entropy);

        Ok(Engine {
            state_ids,
            states,
            rng,
            codebook: md.codebook,
        })
    }
}

impl TryFrom<Metadata> for Oracle {
    type Error = DataFieldNoneError;
    fn try_from(mut md: Metadata) -> Result<Oracle, Self::Error> {
        let data = md.data.ok_or_else(|| DataFieldNoneError)?;

        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: EmptyState = dl_state.into();
                empty_state.0
            })
            .collect();

        Ok(Oracle {
            data,
            states,
            codebook: md.codebook,
        })
    }
}

impl From<DatalessOracle> for Metadata {
    fn from(mut oracle: DatalessOracle) -> Metadata {
        Metadata {
            states: oracle.states.drain(..).map(|state| state.into()).collect(),
            state_ids: None,
            codebook: oracle.codebook,
            data: None,
            rng: None,
        }
    }
}

impl From<Metadata> for DatalessOracle {
    fn from(mut md: Metadata) -> DatalessOracle {
        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: EmptyState = dl_state.into();
                empty_state.0
            })
            .collect();

        DatalessOracle {
            states,
            codebook: md.codebook,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::Example;
    use crate::AppendStrategy;

    #[test]
    fn serde_engine() {
        let engine_1 = Example::Animals.engine().unwrap();
        let serialized_1 = serde_yaml::to_string(&engine_1).unwrap();
        let engine_2: Engine =
            serde_yaml::from_str(serialized_1.as_str()).unwrap();
        let serialized_2 = serde_yaml::to_string(&engine_2).unwrap();

        assert_eq!(serialized_1, serialized_2);
    }

    #[test]
    fn engine_can_run_after_serde() {
        let engine_1 = Example::Animals.engine().unwrap();
        let serialized_1 = serde_yaml::to_string(&engine_1).unwrap();
        let mut engine_2: Engine =
            serde_yaml::from_str(serialized_1.as_str()).unwrap();

        engine_2.run(10);
    }

    #[test]
    fn engine_can_update_data_after() {
        use crate::{InsertMode, OverwriteMode, Row, Value, WriteMode};
        use braid_stats::Datum;

        let engine_1 = Example::Animals.engine().unwrap();
        let serialized_1 = serde_yaml::to_string(&engine_1).unwrap();
        let mut engine_2: Engine =
            serde_yaml::from_str(serialized_1.as_str()).unwrap();

        let rows = vec![Row {
            row_name: "wolf".into(),
            values: vec![Value {
                col_name: "swims".into(),
                value: Datum::Categorical(1),
            }],
        }];

        let res = engine_2.insert_data(
            rows,
            None,
            None,
            WriteMode {
                insert: InsertMode::DenyNewRowsAndColumns,
                overwrite: OverwriteMode::Allow,
                allow_extend_support: false,
                append_strategy: AppendStrategy::None,
            },
        );

        assert!(res.is_ok());
    }

    #[test]
    fn serde_oracle() {
        let oracle_1 = Example::Animals.oracle().unwrap();
        let serialized_1 = serde_yaml::to_string(&oracle_1).unwrap();
        let oracle_2: Oracle =
            serde_yaml::from_str(serialized_1.as_str()).unwrap();
        let serialized_2 = serde_yaml::to_string(&oracle_2).unwrap();

        assert_eq!(serialized_1, serialized_2);
    }

    #[test]
    fn engine_and_oracle_serde_the_same() {
        let engine = Example::Animals.engine().unwrap();
        let serialized = serde_yaml::to_string(&engine).unwrap();
        let _o: Oracle = serde_yaml::from_str(serialized.as_str()).unwrap();
    }
}
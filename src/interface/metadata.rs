use crate::cc::state::StateDiagnostics;
use crate::cc::{
    Assignment, ColModel, Column, ConjugateComponent, DataStore, FeatureData,
    State, View,
};
use crate::dist::{BraidDatum, BraidLikelihood, BraidPrior, BraidStat};
use crate::{DatalessOracle, Engine, Oracle};
use braid_data::SparseContainer;
use braid_stats::labeler::{Label, Labeler, LabelerPrior, LabelerSuffStat};
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use braid_stats::MixtureType;
use once_cell::sync::OnceCell;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::data::{CategoricalSuffStat, GaussianSuffStat, PoissonSuffStat};
use rv::dist::{
    Categorical, Gamma, Gaussian, Mixture, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use rv::traits::ConjugatePrior;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::BTreeMap;
use std::convert::TryFrom;
use thiserror::Error;

use braid_metadata::latest;

impl Into<latest::DatalessState> for State {
    fn into(mut self) -> latest::DatalessState {
        latest::DatalessState {
            views: self.views.drain(..).map(|view| view.into()).collect(),
            asgn: self.asgn.into(),
            weights: self.weights,
            view_alpha_prior: self.view_alpha_prior,
            loglike: self.loglike,
            log_prior: self.log_prior,
            log_view_alpha_prior: self.log_view_alpha_prior,
            log_state_alpha_prior: self.log_state_alpha_prior,
            diagnostics: self.diagnostics.into(),
        }
    }
}

impl Into<latest::DatalessView> for View {
    fn into(mut self) -> latest::DatalessView {
        latest::DatalessView {
            ftrs: {
                let keys: Vec<usize> = self.ftrs.keys().cloned().collect();
                keys.iter()
                    .map(|k| (*k, self.ftrs.remove(k).unwrap().into()))
                    .collect()
            },
            asgn: self.asgn.into(),
            weights: self.weights,
        }
    }
}

impl From<latest::StateDiagnostics> for StateDiagnostics {
    fn from(diag: latest::StateDiagnostics) -> StateDiagnostics {
        StateDiagnostics {
            loglike: diag.loglike,
            log_prior: diag.log_prior,
            nviews: diag.nviews,
            state_alpha: diag.state_alpha,
            ncats_min: diag.ncats_min,
            ncats_max: diag.ncats_max,
            ncats_median: diag.ncats_median,
        }
    }
}

impl From<StateDiagnostics> for latest::StateDiagnostics {
    fn from(diag: StateDiagnostics) -> latest::StateDiagnostics {
        latest::StateDiagnostics {
            loglike: diag.loglike,
            log_prior: diag.log_prior,
            nviews: diag.nviews,
            state_alpha: diag.state_alpha,
            ncats_min: diag.ncats_min,
            ncats_max: diag.ncats_max,
            ncats_median: diag.ncats_median,
        }
    }
}

impl From<latest::Assignment> for Assignment {
    fn from(asgn: latest::Assignment) -> Assignment {
        Assignment {
            alpha: asgn.alpha,
            asgn: asgn.asgn,
            counts: asgn.counts,
            ncats: asgn.ncats,
            prior: asgn.prior,
        }
    }
}

impl From<Assignment> for latest::Assignment {
    fn from(asgn: Assignment) -> latest::Assignment {
        latest::Assignment {
            alpha: asgn.alpha,
            asgn: asgn.asgn,
            counts: asgn.counts,
            ncats: asgn.ncats,
            prior: asgn.prior,
        }
    }
}

macro_rules! col2dataless {
    ($x:ty, $fx:ty, $pr:ty, $h:ty, $s:ty) => {
        impl From<Column<$x, $fx, $pr, $h>>
            for latest::DatalessColumn<$fx, $pr, $h, $s>
        {
            fn from(mut col: Column<$x, $fx, $pr, $h>) -> Self {
                latest::DatalessColumn {
                    id: col.id,
                    components: col
                        .components
                        .drain(..)
                        .map(|cpnt| cpnt.into())
                        .collect(),
                    prior: col.prior,
                    hyper: col.hyper,
                    ignore_hyper: col.ignore_hyper,
                }
            }
        }
    };
}

col2dataless!(
    f64,
    Gaussian,
    NormalInvChiSquared,
    NixHyper,
    GaussianSuffStat
);
col2dataless!(
    u8,
    Categorical,
    SymmetricDirichlet,
    CsdHyper,
    CategoricalSuffStat
);
col2dataless!(Label, Labeler, LabelerPrior, (), LabelerSuffStat);
col2dataless!(u32, Poisson, Gamma, PgHyper, PoissonSuffStat);

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
    ($x:ty, $fx:ty, $pr:ty, $h:ty, $s:ty) => {
        impl Into<EmptyColumn<$x, $fx, $pr, $h>>
            for latest::DatalessColumn<$fx, $pr, $h, $s>
        {
            fn into(mut self) -> EmptyColumn<$x, $fx, $pr, $h> {
                EmptyColumn(Column {
                    id: self.id,
                    components: self
                        .components
                        .drain(..)
                        .map(|cpnt| cpnt.into())
                        .collect(),
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

impl<X, Fx, Pr> From<latest::ConjugateComponent<Fx, Fx::Stat>>
    for ConjugateComponent<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn from(
        cpnt: latest::ConjugateComponent<Fx, Fx::Stat>,
    ) -> ConjugateComponent<X, Fx, Pr> {
        ConjugateComponent {
            fx: cpnt.fx,
            stat: cpnt.stat,
            ln_pp_cache: OnceCell::new(),
        }
    }
}

impl<X, Fx, Pr> From<ConjugateComponent<X, Fx, Pr>>
    for latest::ConjugateComponent<Fx, Fx::Stat>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Fx::Stat: BraidStat,
    Pr: ConjugatePrior<X, Fx>,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    fn from(
        cpnt: ConjugateComponent<X, Fx, Pr>,
    ) -> latest::ConjugateComponent<Fx, Fx::Stat> {
        latest::ConjugateComponent {
            fx: cpnt.fx,
            stat: cpnt.stat,
        }
    }
}

dataless2col!(
    f64,
    Gaussian,
    NormalInvChiSquared,
    NixHyper,
    GaussianSuffStat
);
dataless2col!(
    u8,
    Categorical,
    SymmetricDirichlet,
    CsdHyper,
    CategoricalSuffStat
);
dataless2col!(Label, Labeler, LabelerPrior, (), LabelerSuffStat);
dataless2col!(u32, Poisson, Gamma, PgHyper, PoissonSuffStat);

impl Into<latest::DatalessColModel> for ColModel {
    fn into(self) -> latest::DatalessColModel {
        match self {
            ColModel::Categorical(cm) => {
                latest::DatalessColModel::Categorical(cm.into())
            }
            ColModel::Continuous(cm) => {
                latest::DatalessColModel::Continuous(cm.into())
            }
            ColModel::Labeler(cm) => {
                latest::DatalessColModel::Labeler(cm.into())
            }
            ColModel::Count(cm) => latest::DatalessColModel::Count(cm.into()),
        }
    }
}

impl From<Engine> for latest::Metadata {
    fn from(mut engine: Engine) -> latest::Metadata {
        let data = DataStore(engine.states[0].take_data());
        latest::Metadata {
            states: engine.states.drain(..).map(|state| state.into()).collect(),
            state_ids: Some(engine.state_ids),
            codebook: engine.codebook,
            rng: Some(engine.rng),
            data: Some(data.into()),
        }
    }
}

struct EmptyState(State);

impl From<latest::DatalessState> for EmptyState {
    fn from(mut dl_state: latest::DatalessState) -> EmptyState {
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
                            latest::DatalessColModel::Continuous(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Continuous(ecm.0)
                            }
                            latest::DatalessColModel::Categorical(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Categorical(ecm.0)
                            }
                            latest::DatalessColModel::Labeler(cm) => {
                                let ecm: EmptyColumn<_, _, _, _> = cm.into();
                                ColModel::Labeler(ecm.0)
                            }
                            latest::DatalessColModel::Count(cm) => {
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

#[derive(Clone, Copy, Debug, Error)]
#[error("Cannot deserialize with data field `None`")]
pub struct DataFieldNoneError;

impl From<Oracle> for latest::Metadata {
    fn from(mut oracle: Oracle) -> latest::Metadata {
        latest::Metadata {
            states: oracle.states.drain(..).map(|state| state.into()).collect(),
            state_ids: None,
            codebook: oracle.codebook,
            data: Some(oracle.data.into()),
            rng: None,
        }
    }
}

impl TryFrom<latest::Metadata> for Engine {
    type Error = DataFieldNoneError;
    fn try_from(mut md: latest::Metadata) -> Result<Engine, Self::Error> {
        let data: BTreeMap<usize, FeatureData> = md
            .data
            .ok_or_else(|| DataFieldNoneError)?
            .0
            .drain_filter(|_, _| true)
            .map(|(k, v)| (k, v.into()))
            .collect();

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

impl TryFrom<latest::Metadata> for Oracle {
    type Error = DataFieldNoneError;
    fn try_from(mut md: latest::Metadata) -> Result<Oracle, Self::Error> {
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
            data: data.into(),
            states,
            codebook: md.codebook,
        })
    }
}

impl From<latest::DataStore> for DataStore {
    fn from(mut data: latest::DataStore) -> DataStore {
        let data = data
            .0
            .drain_filter(|_, _| true)
            .map(|(k, v)| (k, v.into()))
            .collect();
        DataStore(data)
    }
}

impl From<DataStore> for latest::DataStore {
    fn from(mut data: DataStore) -> latest::DataStore {
        let data = data
            .0
            .drain_filter(|_, _| true)
            .map(|(k, v)| (k, v.into()))
            .collect();
        latest::DataStore(data)
    }
}

impl From<latest::FeatureData> for FeatureData {
    fn from(data: latest::FeatureData) -> FeatureData {
        match data {
            latest::FeatureData::Continuous(xs) => FeatureData::Continuous(xs),
            latest::FeatureData::Categorical(xs) => {
                FeatureData::Categorical(xs)
            }
            latest::FeatureData::Count(xs) => FeatureData::Count(xs),
            latest::FeatureData::Labeler(xs) => FeatureData::Labeler(xs),
        }
    }
}

impl From<FeatureData> for latest::FeatureData {
    fn from(data: FeatureData) -> latest::FeatureData {
        match data {
            FeatureData::Continuous(xs) => latest::FeatureData::Continuous(xs),
            FeatureData::Categorical(xs) => {
                latest::FeatureData::Categorical(xs)
            }
            FeatureData::Count(xs) => latest::FeatureData::Count(xs),
            FeatureData::Labeler(xs) => latest::FeatureData::Labeler(xs),
        }
    }
}

impl From<DatalessOracle> for latest::Metadata {
    fn from(mut oracle: DatalessOracle) -> latest::Metadata {
        latest::Metadata {
            states: oracle.states.drain(..).map(|state| state.into()).collect(),
            state_ids: None,
            codebook: oracle.codebook,
            data: None,
            rng: None,
        }
    }
}

impl From<latest::Metadata> for DatalessOracle {
    fn from(mut md: latest::Metadata) -> DatalessOracle {
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

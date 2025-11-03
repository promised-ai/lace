use std::convert::TryFrom;

use lace_cc::state::State;
use lace_data::DataStore;
use lace_metadata::latest;
use lace_stats::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use thiserror::Error;

use crate::{DatalessOracle, Engine, Oracle};

impl From<Engine> for latest::Metadata {
    fn from(mut engine: Engine) -> Self {
        let data = DataStore(engine.states[0].take_data());
        Self {
            states: engine.states.drain(..).map(|state| state.into()).collect(),
            state_ids: engine.state_ids,
            codebook: engine.codebook.into(),
            rng: Some(engine.rng),
            data: Some(data.into()),
        }
    }
}

impl From<&Engine> for latest::Metadata {
    fn from(engine: &Engine) -> Self {
        let data = DataStore(engine.states[0].clone().take_data());
        Self {
            states: engine
                .states
                .iter()
                .map(|state| state.clone().into())
                .collect(),
            state_ids: engine.state_ids.clone(),
            codebook: engine.codebook.clone().into(),
            rng: Some(engine.rng.clone()),
            data: Some(data.into()),
        }
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("Failed to convert metadata to Engine/Oracle because `data` field is `None`")]
pub struct DataFieldNoneError;

impl From<Oracle> for latest::Metadata {
    fn from(mut oracle: Oracle) -> Self {
        let n_states = oracle.states.len();
        Self {
            states: oracle.states.drain(..).map(|state| state.into()).collect(),
            state_ids: (0..n_states).collect(),
            codebook: oracle.codebook.into(),
            data: Some(oracle.data.into()),
            rng: None,
        }
    }
}

impl From<&Oracle> for latest::Metadata {
    fn from(oracle: &Oracle) -> Self {
        let n_states = oracle.states.len();
        Self {
            states: oracle
                .states
                .iter()
                .map(|state| state.clone().into())
                .collect(),
            state_ids: (0..n_states).collect(),
            codebook: oracle.codebook.clone().into(),
            data: Some(oracle.data.clone().into()),
            rng: None,
        }
    }
}

impl TryFrom<latest::Metadata> for Engine {
    type Error = DataFieldNoneError;
    fn try_from(mut md: latest::Metadata) -> Result<Self, Self::Error> {
        let data: DataStore = md.data.take().ok_or(DataFieldNoneError)?.into();

        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: latest::EmptyState = dl_state.into();
                let mut state = empty_state.0;
                state.repop_data(data.0.clone());
                state
            })
            .collect();

        let rng = md.rng.unwrap_or_else(Xoshiro256Plus::from_os_rng);

        Ok(Self {
            state_ids: md.state_ids,
            states,
            rng,
            codebook: md.codebook.into(),
        })
    }
}

impl TryFrom<latest::Metadata> for Oracle {
    type Error = DataFieldNoneError;
    fn try_from(mut md: latest::Metadata) -> Result<Self, Self::Error> {
        let data: DataStore = md.data.ok_or(DataFieldNoneError)?.into();

        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: latest::EmptyState = dl_state.into();
                empty_state.0
            })
            .collect();

        Ok(Self {
            data,
            states,
            codebook: md.codebook.into(),
        })
    }
}

impl From<DatalessOracle> for latest::Metadata {
    fn from(mut oracle: DatalessOracle) -> Self {
        let n_states = oracle.states.len();
        Self {
            states: oracle.states.drain(..).map(|state| state.into()).collect(),
            state_ids: (0..n_states).collect(),
            codebook: oracle.codebook.into(),
            data: None,
            rng: None,
        }
    }
}

impl From<latest::Metadata> for DatalessOracle {
    fn from(mut md: latest::Metadata) -> Self {
        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: latest::EmptyState = dl_state.into();
                empty_state.0
            })
            .collect();

        Self {
            states,
            codebook: md.codebook.into(),
        }
    }
}

#[cfg(all(test, feature = "examples"))]
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

        engine_2.run(10).unwrap();
    }

    #[test]
    fn engine_can_update_data_after() {
        use crate::{InsertMode, OverwriteMode, Row, Value, WriteMode};
        use lace_data::Datum;

        let engine_1 = Example::Animals.engine().unwrap();
        let serialized_1 = serde_yaml::to_string(&engine_1).unwrap();
        let mut engine_2: Engine =
            serde_yaml::from_str(serialized_1.as_str()).unwrap();

        let rows = vec![Row::<String, String> {
            row_ix: "wolf".into(),
            values: vec![Value {
                col_ix: "swims".into(),
                value: Datum::Categorical(1_u32.into()),
            }],
        }];

        let res = engine_2.insert_data(
            rows,
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

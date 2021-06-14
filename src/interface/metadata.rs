use std::collections::BTreeMap;
use std::convert::TryFrom;

use braid_cc::state::State;
use braid_data::{DataStore, FeatureData};
use braid_metadata::latest;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use thiserror::Error;

use crate::{DatalessOracle, Engine, Oracle};

impl From<Engine> for latest::Metadata {
    fn from(mut engine: Engine) -> latest::Metadata {
        let data = DataStore(engine.states[0].take_data());
        latest::Metadata {
            states: engine.states.drain(..).map(|state| state.into()).collect(),
            state_ids: Some(engine.state_ids),
            codebook: engine.codebook,
            rng: Some(engine.rng),
            data: Some(data),
        }
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
            data: Some(oracle.data),
            rng: None,
        }
    }
}

impl TryFrom<latest::Metadata> for Engine {
    type Error = DataFieldNoneError;
    fn try_from(mut md: latest::Metadata) -> Result<Engine, Self::Error> {
        let data: BTreeMap<usize, FeatureData> =
            md.data.take().ok_or(DataFieldNoneError)?.0;

        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: latest::EmptyState = dl_state.into();
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
        let data = md.data.ok_or(DataFieldNoneError)?;

        let states: Vec<State> = md
            .states
            .drain(..)
            .map(|dl_state| {
                let empty_state: latest::EmptyState = dl_state.into();
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
                let empty_state: latest::EmptyState = dl_state.into();
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
        use braid_data::Datum;

        let engine_1 = Example::Animals.engine().unwrap();
        let serialized_1 = serde_yaml::to_string(&engine_1).unwrap();
        let mut engine_2: Engine =
            serde_yaml::from_str(serialized_1.as_str()).unwrap();

        let rows = vec![Row {
            row_ix: "wolf".into(),
            values: vec![Value {
                col_ix: "swims".into(),
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

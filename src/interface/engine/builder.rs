use braid_codebook::codebook::Codebook;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use super::error::NewEngineError;
use super::Engine;
use crate::data::DataSource;

const DEFAULT_NSTATES: usize = 8;
const DEFAULT_ID_OFFSET: usize = 0;

/// Builds `Engine`s
pub struct EngineBuilder {
    nstates: Option<usize>,
    codebook: Option<Codebook>,
    data_source: DataSource,
    id_offset: Option<usize>,
    seed: Option<u64>,
}

impl EngineBuilder {
    pub fn new(data_source: DataSource) -> Self {
        EngineBuilder {
            nstates: None,
            codebook: None,
            data_source,
            id_offset: None,
            seed: None,
        }
    }

    /// Eith a certain number of states
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.nstates = Some(nstates);
        self
    }

    /// With a specific codebook
    pub fn with_codebook(mut self, codebook: Codebook) -> Self {
        self.codebook = Some(codebook);
        self
    }

    /// With state IDs starting at an offset
    pub fn with_id_offset(mut self, id_offset: usize) -> Self {
        self.id_offset = Some(id_offset);
        self
    }

    /// With a given random number generator
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // Build the `Engine`; consume the `EngineBuilder`.
    pub fn build(self) -> Result<Engine, NewEngineError> {
        let nstates = self.nstates.unwrap_or(DEFAULT_NSTATES);

        let id_offset = self.id_offset.unwrap_or(DEFAULT_ID_OFFSET);
        let rng = match self.seed {
            Some(s) => Xoshiro256Plus::seed_from_u64(s),
            None => Xoshiro256Plus::from_entropy(),
        };

        // FIXME-RESULT
        let codebook = self
            .codebook
            .unwrap_or(self.data_source.default_codebook().unwrap());

        Engine::new(nstates, codebook, self.data_source, id_offset, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maplit::btreeset;
    use std::{collections::BTreeSet, path::PathBuf};

    fn animals_csv() -> DataSource {
        DataSource::Csv(PathBuf::from("resources/datasets/animals/data.csv"))
    }

    #[test]
    fn default_build_settings() {
        let engine = EngineBuilder::new(animals_csv()).build().unwrap();
        let state_ids: BTreeSet<usize> =
            engine.states.keys().map(|k| *k).collect();
        let target_ids: BTreeSet<usize> = btreeset! {0, 1, 2, 3, 4, 5, 6, 7};
        assert_eq!(engine.nstates(), 8);
        assert_eq!(state_ids, target_ids);
    }

    #[test]
    fn with_id_offet_3() {
        let engine = EngineBuilder::new(animals_csv())
            .with_id_offset(3)
            .build()
            .unwrap();
        let state_ids: BTreeSet<usize> =
            engine.states.keys().map(|k| *k).collect();
        let target_ids: BTreeSet<usize> = btreeset! {3, 4, 5, 6, 7, 8, 9, 10};
        assert_eq!(engine.nstates(), 8);
        assert_eq!(state_ids, target_ids);
    }

    #[test]
    fn with_nstates_3() {
        let engine = EngineBuilder::new(animals_csv())
            .with_nstates(3)
            .build()
            .unwrap();
        let state_ids: BTreeSet<usize> =
            engine.states.keys().map(|k| *k).collect();
        let target_ids: BTreeSet<usize> = btreeset! {0, 1, 2};
        assert_eq!(engine.nstates(), 3);
        assert_eq!(state_ids, target_ids);
    }

    #[test]
    fn with_nstates_0_causes_error() {
        let result = EngineBuilder::new(animals_csv()).with_nstates(0).build();

        assert!(result.is_err());
    }

    // FIXME: Seed control is not working
    #[test]
    #[ignore]
    fn seeding_works_single_states() {
        let mut engine_1 = EngineBuilder::new(animals_csv())
            .with_nstates(1)
            .with_seed(8675309)
            .build()
            .unwrap();

        engine_1.run(10);

        let mut engine_2 = EngineBuilder::new(animals_csv())
            .with_nstates(1)
            .with_seed(8675309)
            .build()
            .unwrap();

        engine_2.run(10);

        let asgn_1 = &engine_1.states.get(&0).unwrap().asgn;
        let asgn_2 = &engine_2.states.get(&0).unwrap().asgn;
        assert_eq!(asgn_1, asgn_2);
    }
}

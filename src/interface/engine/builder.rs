use braid_codebook::Codebook;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use thiserror::Error;

use super::error::NewEngineError;
use super::Engine;
use crate::data::{DataSource, DefaultCodebookError};

const DEFAULT_NSTATES: usize = 8;
const DEFAULT_ID_OFFSET: usize = 0;

/// Builds `Engine`s
pub struct EngineBuilder {
    nstates: Option<usize>,
    codebook: Option<Codebook>,
    data_source: DataSource,
    id_offset: Option<usize>,
    seed: Option<u64>,
    flat_cols: bool,
}

#[derive(Debug, Error)]
pub enum BuildEngineError {
    #[error("error constructing Engine: {0}")]
    NewEngineError(#[from] NewEngineError),
    #[error("error generating default codebook: {0}")]
    DefaultCodebookError(#[from] DefaultCodebookError),
}

impl EngineBuilder {
    #[must_use]
    pub fn new(data_source: DataSource) -> Self {
        Self {
            nstates: None,
            codebook: None,
            data_source,
            id_offset: None,
            seed: None,
            flat_cols: false,
        }
    }

    /// Eith a certain number of states
    #[must_use]
    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.nstates = Some(nstates);
        self
    }

    /// With a specific codebook
    #[must_use]
    pub fn with_codebook(mut self, codebook: Codebook) -> Self {
        self.codebook = Some(codebook);
        self
    }

    /// With state IDs starting at an offset
    #[must_use]
    pub fn with_id_offset(mut self, id_offset: usize) -> Self {
        self.id_offset = Some(id_offset);
        self
    }

    /// With a given random number generator
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// With a flat column structure -- one view in each state
    #[must_use]
    pub fn flat_cols(mut self) -> Self {
        self.flat_cols = true;
        self
    }

    // Build the `Engine`; consume the `EngineBuilder`.
    pub fn build(self) -> Result<Engine, BuildEngineError> {
        let nstates = self.nstates.unwrap_or(DEFAULT_NSTATES);

        let id_offset = self.id_offset.unwrap_or(DEFAULT_ID_OFFSET);
        let rng = match self.seed {
            Some(s) => Xoshiro256Plus::seed_from_u64(s),
            None => Xoshiro256Plus::from_entropy(),
        };

        let codebook = match self.codebook {
            Some(codebook) => Ok(codebook),
            None => self
                .data_source
                .default_codebook()
                .map_err(BuildEngineError::DefaultCodebookError),
        }?;

        let mut engine =
            Engine::new(nstates, codebook, self.data_source, id_offset, rng)
                .map_err(BuildEngineError::NewEngineError)?;

        engine.flatten_cols();

        Ok(engine)
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
            engine.state_ids.iter().copied().collect();
        let target_ids: BTreeSet<usize> = btreeset! {0, 1, 2, 3, 4, 5, 6, 7};
        assert_eq!(engine.nstates(), 8);
        assert_eq!(state_ids, target_ids);
    }

    #[test]
    fn gzipped_csv() {
        let path = PathBuf::from("resources/datasets/animals/data.csv.gz");
        let datasource = DataSource::GzipCsv(path);
        let engine = EngineBuilder::new(datasource).build().unwrap();

        let state_ids: BTreeSet<usize> =
            engine.state_ids.iter().copied().collect();
        let target_ids: BTreeSet<usize> = btreeset! {0, 1, 2, 3, 4, 5, 6, 7};
        assert_eq!(engine.nstates(), 8);
        assert_eq!(state_ids, target_ids);

        engine.run(10);
    }

    #[test]
    fn with_id_offet_3() {
        let engine = EngineBuilder::new(animals_csv())
            .with_id_offset(3)
            .build()
            .unwrap();
        let state_ids: BTreeSet<usize> =
            engine.state_ids.iter().copied().collect();
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
            engine.state_ids.iter().copied().collect();
        let target_ids: BTreeSet<usize> = btreeset! {0, 1, 2};
        assert_eq!(engine.nstates(), 3);
        assert_eq!(state_ids, target_ids);
    }

    #[test]
    fn with_nstates_0_causes_error() {
        let result = EngineBuilder::new(animals_csv()).with_nstates(0).build();

        assert!(result.is_err());
    }

    #[test]
    fn seeding_engine_works() {
        let seed: u64 = 8_675_309;
        let nstates = 4;
        let mut engine_1 = EngineBuilder::new(animals_csv())
            .with_nstates(nstates)
            .with_seed(seed)
            .build()
            .unwrap();

        let mut engine_2 = EngineBuilder::new(animals_csv())
            .with_nstates(nstates)
            .with_seed(seed)
            .build()
            .unwrap();

        // initial state should be the same
        for (state_1, state_2) in
            engine_1.states.iter().zip(engine_2.states.iter())
        {
            assert_eq!(&state_1.asgn, &state_2.asgn);
            for (view_1, view_2) in
                state_1.views.iter().zip(state_2.views.iter())
            {
                assert_eq!(&view_1.asgn, &view_2.asgn);
            }
        }

        engine_1.run(10);
        engine_2.run(10);

        // And should stay the same after the run
        for (state_1, state_2) in
            engine_1.states.iter().zip(engine_2.states.iter())
        {
            assert_eq!(&state_1.asgn, &state_2.asgn);
            for (view_1, view_2) in
                state_1.views.iter().zip(state_2.views.iter())
            {
                assert_eq!(&view_1.asgn, &view_2.asgn);
            }
        }
    }
}

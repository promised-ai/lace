use braid_codebook::codebook::Codebook;
use rand::{FromEntropy, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use crate::data::DataSource;
use crate::interface::Engine;
use crate::result;

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
    pub fn build(self) -> result::Result<Engine> {
        let nstates = self.nstates.unwrap_or(DEFAULT_NSTATES);
        let id_offset = self.id_offset.unwrap_or(DEFAULT_ID_OFFSET);
        let rng = match self.seed {
            Some(s) => Xoshiro256Plus::seed_from_u64(s),
            None => Xoshiro256Plus::from_entropy(),
        };

        let codebook = self
            .codebook
            .unwrap_or(self.data_source.default_codebook()?);

        Ok(Engine::new(
            nstates,
            codebook,
            self.data_source,
            id_offset,
            rng,
        ))
    }
}

// FIXME: tests!

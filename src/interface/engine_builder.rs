extern crate braid_codebook;
extern crate rand;

use braid_codebook::codebook::Codebook;
use rand::{FromEntropy, XorShiftRng};

use crate::data::DataSource;
use crate::interface::Engine;
use crate::result;

/// Builds `Engine`s
pub struct EngineBuilder {
    nstates: Option<usize>,
    codebook: Option<Codebook>,
    data_source: DataSource,
    id_offset: Option<usize>,
    rng: Option<XorShiftRng>,
}

impl EngineBuilder {
    pub fn new(data_source: DataSource) -> Self {
        EngineBuilder {
            nstates: None,
            codebook: None,
            data_source,
            id_offset: None,
            rng: None,
        }
    }

    pub fn with_nstates(mut self, nstates: usize) -> Self {
        self.nstates = Some(nstates);
        self
    }

    pub fn with_codebook(mut self, codebook: Codebook) -> Self {
        self.codebook = Some(codebook);
        self
    }

    pub fn with_id_offset(mut self, id_offset: usize) -> Self {
        self.id_offset = Some(id_offset);
        self
    }

    pub fn with_rng(mut self, rng: XorShiftRng) -> Self {
        self.rng = Some(rng);
        self
    }

    // Build the `Engine`; consume the `EngineBuilder`.
    pub fn build(self) -> result::Result<Engine> {
        let nstates = self.nstates.unwrap_or(8);
        let id_offset = self.id_offset.unwrap_or(0);
        let rng = self.rng.unwrap_or(XorShiftRng::from_entropy());
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

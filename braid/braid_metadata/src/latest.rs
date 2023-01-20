use crate::versions::{v1, v2};
use crate::{impl_metadata_version, to_from_newtype, MetadataVersion};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};

// re-exports used by braid lib
pub use v2::{DatalessState, EmptyState};

pub const METADATA_VERSION: u32 = 3;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Codebook(pub braid_codebook::Codebook);

to_from_newtype!(braid_codebook::Codebook, Codebook);

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub states: Vec<v2::DatalessState>,
    pub state_ids: Vec<usize>,
    pub codebook: Codebook,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<v1::DataStore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng: Option<Xoshiro256Plus>,
}

impl_metadata_version!(Metadata, METADATA_VERSION);
impl_metadata_version!(Codebook, METADATA_VERSION);

// Create the loaders module for latest
crate::loaders!(
    v2::DatalessState,
    v1::DataStore,
    Codebook,
    rand_xoshiro::Xoshiro256Plus
);

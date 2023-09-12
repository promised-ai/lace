#![warn(unused_extern_crates)]
#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]

mod config;
mod error;
pub mod latest;
mod utils;
pub mod versions;

pub use utils::{deserialize_file, save_state, serialize_obj};

use log::info;
use std::path::Path;

pub use config::{FileConfig, SerializedType};
pub use error::Error;

pub trait MetadataVersion {
    fn metadata_version() -> i32;
}

/// Implements the MetadataVersion trait
#[macro_export]
macro_rules! impl_metadata_version {
    ($type:ty, $version:expr) => {
        impl MetadataVersion for $type {
            fn metadata_version() -> i32 {
                $version
            }
        }
    };
}

/// For a newtype `Outer(Inner)`, implements `From<Inner>` for `Outer` and
/// `From<Outer>` for `Inner`.
#[macro_export]
macro_rules! to_from_newtype {
    ($from:ty, $to:ty) => {
        impl From<$from> for $to {
            fn from(x: $from) -> Self {
                Self(x)
            }
        }

        impl From<$to> for $from {
            fn from(x: $to) -> Self {
                x.0
            }
        }
    };
}

/// creates a bunch of helper functions in a `load` module that load the
/// metadata components and create and `Meatadata` object of the appropriate
/// version.
#[macro_export]
macro_rules! loaders {
    ($state:ident, $data:ty, $codebook:ty, $rng:ty) => {
        pub(crate) mod load {
            use super::*;
            use log::info;
            use rayon::prelude::*;
            use std::path::Path;
            use $crate::config::FileConfig;
            use $crate::utils::{
                get_codebook_path, get_data_path, get_rng_path, get_state_ids,
                get_state_path, load, load_as_type, read_diagnostics,
            };
            use $crate::{Error, SerializedType};

            pub(crate) fn load_rng<P: AsRef<Path>>(
                path: P,
            ) -> Result<$rng, Error> {
                let rng_path = get_rng_path(path);
                info!("Loading RNG at {:?}...", rng_path);
                load_as_type(rng_path, SerializedType::Yaml)
            }

            pub(crate) fn load_state<P: AsRef<Path>>(
                path: P,
                state_id: usize,
                file_config: &FileConfig,
            ) -> Result<$state, Error> {
                let state_path = get_state_path(path.as_ref(), state_id);
                info!("Loading state at {:?}...", state_path);
                let serialized_type = file_config.serialized_type;
                let state: DatalessState =
                    load(state_path.as_path(), serialized_type)?;
                let diagnostics = read_diagnostics(path.as_ref(), state_id)?;
                Ok($state { state, diagnostics })
            }

            /// Return (states, state_ids) tuple
            pub(crate) fn load_states<P: AsRef<Path>>(
                path: P,
                file_config: &FileConfig,
            ) -> Result<(Vec<$state>, Vec<usize>), Error> {
                let path = path.as_ref();
                let state_ids = get_state_ids(path)?;
                let states: Result<Vec<_>, Error> = state_ids
                    .par_iter()
                    .map(|&id| load_state(path, id, file_config))
                    .collect();

                info!("States loaded");
                states.map(|s| (s, state_ids))
            }

            pub(crate) fn load_data<P: AsRef<Path>>(
                path: P,
                file_config: &FileConfig,
            ) -> Result<$data, Error> {
                let data_path = get_data_path(path);
                info!("Loading data at {:?}...", data_path);
                let data: $data = load(data_path, file_config.serialized_type)?;
                Ok(data)
            }

            pub(crate) fn load_codebook<P: AsRef<Path>>(
                path: P,
                file_config: &FileConfig,
            ) -> Result<$codebook, Error> {
                let codebook_path = get_codebook_path(path);
                info!("Loading codebook at {:?}...", codebook_path);
                load(codebook_path, file_config.serialized_type)
            }

            pub(crate) fn load_meatadata<P: AsRef<std::path::Path>>(
                path: P,
                file_config: &$crate::config::FileConfig,
            ) -> Result<Metadata, $crate::error::Error> {
                let path = path.as_ref();
                // TODO: handle error properly
                // TODO: load component in parallel
                let data = load_data(path, &file_config).ok();
                let (states, state_ids) = load_states(&path, &file_config)?;
                let codebook = load_codebook(path, &file_config)?;
                let rng = load_rng(path).ok();
                Ok(Metadata {
                    states,
                    state_ids,
                    codebook,
                    data,
                    rng,
                })
            }
        }
    };
}

pub fn save_metadata<P: AsRef<Path>>(
    metadata: &latest::Metadata,
    path: P,
    ser_type: SerializedType,
) -> Result<(), Error> {
    let path = path.as_ref();

    utils::path_validator(path)?;
    {
        let file_config = FileConfig {
            metadata_version: latest::METADATA_VERSION,
            serialized_type: ser_type,
            #[cfg(feature = "experimental")]
            experimental: true,
            #[cfg(not(feature = "experimental"))]
            experimental: false,
        };
        utils::save_file_config(path, &file_config)?;
    }

    if let Some(ref data) = metadata.data {
        info!("Saving data to {:?}...", path);
        utils::save_data(path, data, ser_type)?;
    } else {
        info!("Data is None. Skipping.");
    }

    info!("Saving codebook to {:?}...", path);
    utils::save_codebook(path, &metadata.codebook, ser_type)?;

    if let Some(ref rng) = metadata.rng {
        info!("Saving rng to {:?}...", path);
        utils::save_rng(path, rng)?;
    } else {
        info!("RNG is None. Skipping.");
    }

    info!("Saving states to {:?}...", path);
    utils::save_states(path, &metadata.states, &metadata.state_ids, ser_type)
}

pub fn load_metadata<P: AsRef<Path>>(
    path: P,
) -> Result<latest::Metadata, Error> {
    let path = path.as_ref();

    utils::path_validator(path)?;
    let file_config = utils::load_file_config(path)?;

    let md_version = file_config.metadata_version;
    match md_version {
        0 => {
            let v0 =
                crate::versions::v0::load::load_meatadata(path, &file_config)?;
            Ok(v0.into())
        }
        1 => crate::latest::load::load_meatadata(path, &file_config),
        requested => Err(Error::UnsupportedMetadataVersion {
            requested,
            max_supported: crate::latest::METADATA_VERSION,
        }),
    }
}

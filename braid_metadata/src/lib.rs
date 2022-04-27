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
pub mod convert;
mod error;
pub mod latest;
mod utils;
pub mod versions;

use log::info;
use std::path::Path;

pub use config::{
    encryption_key_from_profile, encryption_key_string_from_profile,
    EncryptionKey, FileConfig, SaveConfig, SerializedType, UserInfo,
};
pub use error::Error;

pub trait MetadataVersion {
    fn metadata_version() -> u32;
}

/// Implements the MetadataVersion trait
#[macro_export]
macro_rules! impl_metadata_version {
    ($type:ty, $version:expr) => {
        impl MetadataVersion for $type {
            fn metadata_version() -> u32 {
                $version
            }
        }
    };
}

/// For a newtype Outer(Inner), implements From<Inner> for Outer and From<Outer>
/// for Inner.
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
    ($state:ty, $data:ty, $codebook:ty, $rng:ty) => {
        pub(crate) mod load {
            use super::*;
            use crate::config::FileConfig;
            use crate::utils::{
                get_codebook_path, get_data_path, get_rng_path, get_state_ids,
                get_state_path, load_as_possibly_encrypted, load_as_type,
            };
            use crate::{Error, SerializedType};
            use log::info;
            use std::path::Path;

            pub(crate) fn load_rng<P: AsRef<Path>>(
                path: P,
            ) -> Result<$rng, Error> {
                let rng_path = get_rng_path(path);
                load_as_type(rng_path, SerializedType::Yaml)
            }

            pub(crate) fn load_state<P: AsRef<Path>>(
                path: P,
                state_id: usize,
                file_config: FileConfig,
                key: Option<&[u8; 32]>,
            ) -> Result<$state, Error> {
                let state_path = get_state_path(path, state_id);
                info!("Loading state at {:?}...", state_path);
                let serialized_type = file_config.serialized_type;
                load_as_possibly_encrypted(
                    state_path.as_path(),
                    serialized_type,
                    key,
                )
            }

            /// Return (states, state_ids) tuple
            pub(crate) fn load_states<P: AsRef<Path>>(
                path: P,
                file_config: FileConfig,
                key: Option<&[u8; 32]>,
            ) -> Result<(Vec<$state>, Vec<usize>), Error> {
                let state_ids = get_state_ids(path.as_ref())?;
                let states: Result<Vec<_>, Error> = state_ids
                    .iter()
                    .map(|&id| load_state(path.as_ref(), id, file_config, key))
                    .collect();

                states.map(|s| (s, state_ids))
            }

            pub(crate) fn load_data<P: AsRef<Path>>(
                path: P,
                file_config: FileConfig,
                key: Option<&[u8; 32]>,
            ) -> Result<$data, Error> {
                let data_path = get_data_path(path);
                let data: $data = load_as_possibly_encrypted(
                    data_path,
                    file_config.serialized_type,
                    key,
                )?;
                Ok(data)
            }

            pub(crate) fn load_codebook<P: AsRef<Path>>(
                path: P,
                file_config: FileConfig,
                key: Option<&[u8; 32]>,
            ) -> Result<$codebook, Error> {
                let codebook_path = get_codebook_path(path);
                load_as_possibly_encrypted(
                    codebook_path,
                    file_config.serialized_type,
                    key,
                )
            }

            pub(crate) fn load_meatadata<P: AsRef<std::path::Path>>(
                path: P,
                file_config: crate::config::FileConfig,
                encryption_key: Option<&[u8; 32]>,
            ) -> Result<Metadata, crate::error::Error> {
                let path = path.as_ref();
                let data = load_data(path, file_config, encryption_key).ok();
                let (states, state_ids) =
                    load_states(&path, file_config, encryption_key)?;
                let codebook =
                    load_codebook(path, file_config, encryption_key)?;
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
    mut save_config: SaveConfig,
) -> Result<(), Error> {
    let path = path.as_ref();

    let file_config = save_config.to_file_config();
    let encryption_key = save_config.encryption_key()?;

    utils::path_validator(path)?;
    utils::save_file_config(path, file_config)?;

    if let Some(ref data) = metadata.data {
        info!("Saving data to {:?}...", path);
        utils::save_data(path, data, file_config, encryption_key)?;
    } else {
        info!("Data is None. Skipping.");
    }

    info!("Saving codebook to {:?}...", path);
    utils::save_codebook(
        path,
        &metadata.codebook,
        file_config,
        encryption_key,
    )?;

    if let Some(ref rng) = metadata.rng {
        info!("Saving rng to {:?}...", path);
        utils::save_rng(path, rng)?;
    } else {
        info!("RNG is None. Skipping.");
    }

    info!("Saving states to {:?}...", path);
    utils::save_states(
        path,
        &metadata.states,
        &metadata.state_ids,
        file_config,
        encryption_key,
    )
}

pub fn load_metadata<P: AsRef<Path>>(
    path: P,
    encryption_key: Option<&[u8; 32]>,
) -> Result<latest::Metadata, Error> {
    let path = path.as_ref();

    utils::path_validator(path)?;
    let file_config = utils::load_file_config(path)?;

    let md_version = file_config.metadata_version;
    match md_version {
        1 => {
            info!("Converting metadata v1 format to latest format");
            crate::versions::v1::load::load_meatadata(
                path,
                file_config,
                encryption_key,
            )
            .map(latest::Metadata::from)
        }
        2 => crate::latest::load::load_meatadata(
            path,
            file_config,
            encryption_key,
        ),
        requested => Err(Error::UnsupportedMetadataVersion {
            requested,
            max_supported: crate::latest::METADATA_VERSION,
        }),
    }
}

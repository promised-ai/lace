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

use log::info;
use serde_encrypt::shared_key::SharedKey;
use std::path::Path;

pub use config::{
    encryption_key_from_profile, encryption_key_string_from_profile,
    EncryptionKey, FileConfig, SaveConfig, SerializedType, UserInfo,
};
pub use error::Error;

pub trait MetadataVersion {
    fn metadata_version() -> u32;
}

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
    encryption_key: Option<&SharedKey>,
) -> Result<latest::Metadata, Error> {
    use latest::METADATA_VERSION;
    let path = path.as_ref();

    utils::path_validator(path)?;
    let file_config = utils::load_file_config(path)?;
    if file_config.metadata_version > METADATA_VERSION {
        panic!(
            "{:?} was saved with metadata version {}, but this version \
            of braid only support up to version {}.",
            path, file_config.metadata_version, METADATA_VERSION
        );
    }

    let data = utils::load_data(path, file_config, encryption_key).ok();
    let (states, state_ids) =
        utils::load_states(path, file_config, encryption_key)?;
    let codebook = utils::load_codebook(path, file_config, encryption_key)?;
    let rng = utils::load_rng(path).ok();

    Ok(latest::Metadata {
        states,
        state_ids,
        codebook,
        data,
        rng,
    })
}

use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unable to retrieve the home directory")]
    CouldNotGetHomeDirectory,
    #[error("File not found")]
    FileNotFound,
    #[error(
        "Invalid state file name `{0}`. State file should be names `<N>.state` \
        where <N> is an integer."
    )]
    StateFileNameInvalid(String),
    #[error(
        "Invalid serialized type `{0}`. Options are `bincode`, `yaml`, and \
        `encrypted`."
    )]
    SerializedTypeInvalid(String),
    #[error("IoError: {0}")]
    Io(#[from] io::Error),
    #[error("YamlError: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("JsonError: {0}")]
    Json(#[from] serde_json::Error),
    #[error("BincodeError: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("HexError: {0}")]
    Hex(#[from] hex::FromHexError),
    #[error("Unsupported metadata version `{requested}`. Max supported version: {max_supported}")]
    UnsupportedMetadataVersion { requested: i32, max_supported: i32 },
    #[error("Failure parsing float in diagnostics: {0}")]
    DiagnosticsParseInt(#[from] std::num::ParseIntError),
    #[error("Failure parsing float in diagnostics: {0}")]
    DiagnosticsParseFloat(#[from] std::num::ParseFloatError),
    #[error("Other: {0}")]
    Other(String),
}

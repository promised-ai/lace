use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TomlError {
    #[error("{0}")]
    Ser(#[from] toml::ser::Error),
    #[error("{0}")]
    De(#[from] toml::de::Error),
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unable to retrieve the home directory")]
    CouldNotGetHomeDirectory,
    #[error("File not found")]
    FileNotFound,
    #[error(
        "Encryption key required but was not supplied nor was it found in \
        the `default` profile"
    )]
    EncryptionKeyNotFound,
    #[error("The profile `{0}` was not found")]
    ProfileNotFound(String),
    #[error("No credentials file found in $HOME/.braid/credentials")]
    CredentialsNotFound,
    #[error("Invalid encryption key not found for profile `{0}`")]
    EncryptionKeyNotFoundForProfile(String),
    #[error("Could not convert supplied string into an encryption key")]
    StringIsInvalidEncryptionKey,
    #[error("Could not decrypt with the supplied key")]
    CouldNotDecryptWithKey,
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
    #[error("TomlError: {0}")]
    Toml(#[from] TomlError),
    #[error("JsonError: {0}")]
    Json(#[from] serde_json::Error),
    #[error("BincodeError: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Unspecified crypto error")]
    UnspecifiedCrypto,
    #[error("HexError: {0}")]
    Hex(#[from] hex::FromHexError),
    #[error("Unsupported metadata version `{requested}`. Max supported version: {max_supported}")]
    UnsupportedMetadataVersion { requested: u32, max_supported: u32 },
    #[error("Other: {0}")]
    Other(String),
}

impl From<ring::error::Unspecified> for Error {
    fn from(_err: ring::error::Unspecified) -> Self {
        Self::UnspecifiedCrypto
    }
}

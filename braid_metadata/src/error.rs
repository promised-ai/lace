use std::io;
use thiserror::Error;

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
    SertializedTypeInvalid(String),
    #[error("IoError: {0}")]
    Io(#[from] io::Error),
    #[error("YamlError: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("BincodeError: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("EncryptError: {0}")]
    Encrypt(#[from] serde_encrypt::Error),
    #[error("HexError: {0}")]
    Hex(#[from] hex::FromHexError),
    #[error("Other: {0}")]
    Other(String),
}

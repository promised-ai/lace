use std::convert::{TryFrom, TryInto};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::Error;

/// Denotes `State` file type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializedType {
    /// Fast, binary format
    Bincode,
    /// Slow, human-readable format
    Yaml,
    /// Shared-key encrypted
    Encrypted,
}

impl FromStr for SerializedType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bincode" => Ok(Self::Bincode),
            "yaml" => Ok(Self::Yaml),
            "encrypted" => Ok(Self::Encrypted),
            _ => Err(Self::Err::SertializedTypeInvalid(String::from(s))),
        }
    }
}

impl Default for SerializedType {
    fn default() -> Self {
        Self::Yaml
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub struct EncryptionKey(String);

impl From<String> for EncryptionKey {
    fn from(key_string: String) -> Self {
        EncryptionKey(key_string)
    }
}

impl From<EncryptionKey> for String {
    fn from(encryption_key: EncryptionKey) -> Self {
        encryption_key.0
    }
}

impl From<&[u8]> for EncryptionKey {
    fn from(shared_key: &[u8]) -> Self {
        let key_string = hex::encode(shared_key);
        EncryptionKey(key_string)
    }
}

impl TryFrom<EncryptionKey> for [u8; 32] {
    type Error = Error;

    fn try_from(encryption_key: EncryptionKey) -> Result<Self, Self::Error> {
        let mut array = [0_u8; 32];
        hex::decode_to_slice(encryption_key.0, &mut array)?;
        Ok(array)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct UserProfile {
    encryption_key: Option<EncryptionKey>,
}

impl TryInto<Option<[u8; 32]>> for UserProfile {
    type Error = Error;

    fn try_into(self) -> Result<Option<[u8; 32]>, Self::Error> {
        if self.encryption_key.is_none() {
            return Ok(None);
        }

        let hex_string = self.encryption_key.unwrap();
        let mut array = [0_u8; 32];
        hex::decode_to_slice(String::from(hex_string), &mut array)?;

        Ok(Some(array))
    }
}

#[derive(PartialEq, Eq, Default)]
pub struct UserInfo {
    pub encryption_key: Option<[u8; 32]>,
    pub profile: Option<String>,
}

pub fn encryption_key_string_from_profile(
    profile_name: &str,
) -> Result<Option<String>, Error> {
    use std::collections::HashMap;
    use std::fs::OpenOptions;
    use std::io::Read;

    let mut profiles_path =
        dirs::home_dir().ok_or(Error::CouldNotGetHomeDirectory)?;
    profiles_path.push(".braid");
    profiles_path.push("credentials");

    let mut user_profiles: HashMap<String, UserProfile> = {
        let mut file = OpenOptions::new().read(true).open(profiles_path)?;

        let mut buf: Vec<u8> = Vec::new();
        file.read_exact(&mut buf)?;

        toml::from_slice(&buf).unwrap()
    };

    let key: Option<String> =
        user_profiles.remove(profile_name).and_then(|user_profile| {
            user_profile.encryption_key.map(|key| key.into())
        });

    Ok(key)
}

pub fn encryption_key_from_profile(
    profile_name: &str,
) -> Result<Option<[u8; 32]>, Error> {
    encryption_key_string_from_profile(profile_name)?
        .map(EncryptionKey::from)
        .map(|key| key.try_into())
        .transpose()
}

impl UserInfo {
    /// Return the encryption key
    ///
    /// Only returns the encryption key if it exists or can be retrieved from a
    /// profile.
    ///
    /// If the encryption key is provided, it will be returned as a reference.
    /// If the key is not provided, this function will look in either the
    /// provided profile or the default profile if a profile was not supplied.
    /// If no key is supplied, no profile is supplied, and no key exists in the
    /// default profile, then `None` is returned.
    ///
    /// # Errors
    /// Will return a Error::EncryptionKeyNotFoundForProfile error if the no key
    /// exists for the provided profile.
    pub fn encryption_key(&mut self) -> Result<Option<&[u8; 32]>, Error> {
        if self.encryption_key.is_some() {
            return Ok(self.encryption_key.as_ref());
        }

        let encryption_key = if let Some(ref profile) = self.profile {
            encryption_key_from_profile(profile)?
        } else {
            // FIXME: check for `BRAID_PRPFILE` variables
            encryption_key_from_profile("default").ok().flatten()
        };

        self.encryption_key = encryption_key;

        Ok(self.encryption_key.as_ref())
    }
}

#[derive(PartialEq, Eq)]
pub struct SaveConfig {
    pub metadata_version: u32,
    pub serialized_type: SerializedType,
    pub user_info: UserInfo,
}

impl SaveConfig {
    pub fn encryption_key(&mut self) -> Result<Option<&[u8; 32]>, Error> {
        self.user_info.encryption_key()
    }
}

impl Default for SaveConfig {
    fn default() -> Self {
        Self {
            metadata_version: crate::latest::METADATA_VERSION,
            serialized_type: SerializedType::default(),
            user_info: UserInfo::default(),
        }
    }
}

impl SaveConfig {
    pub fn to_file_config(&self) -> FileConfig {
        FileConfig {
            metadata_version: self.metadata_version,
            serialized_type: self.serialized_type,
        }
    }
}

/// How to load/save the files for an oracle or engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileConfig {
    /// Metadata version
    pub metadata_version: u32,
    pub serialized_type: SerializedType,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            metadata_version: crate::latest::METADATA_VERSION,
            serialized_type: SerializedType::default(),
        }
    }
}

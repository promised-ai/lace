use std::str::FromStr;

use serde::Deserialize;
use serde::Serialize;

use crate::metadata::Error;

/// Denotes `State` file type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SerializedType {
    /// Fast, binary format
    #[default]
    Bincode,
    /// Slow, human-readable format
    Yaml,
    /// Everybody's favorite
    Json,
}

impl FromStr for SerializedType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bincode" => Ok(Self::Bincode),
            "yaml" | "yml" => Ok(Self::Yaml),
            "json" => Ok(Self::Json),
            _ => Err(Self::Err::SerializedTypeInvalid(String::from(s))),
        }
    }
}


#[derive(PartialEq, Eq, Clone, Debug, Serialize, Deserialize)]
pub struct FileConfig {
    pub metadata_version: i32,
    pub serialized_type: SerializedType,
}

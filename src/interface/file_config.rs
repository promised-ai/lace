extern crate serde;

use serde::{Deserialize, Serialize};

pub const CURRENT_FILE_CONFIG_VERSION: u32 = 1;

/// Which type of file is each state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializedType {
    Bincode,
    Yaml,
}

impl Default for SerializedType {
    fn default() -> Self {
        SerializedType::Yaml
    }
}

/// How to load/save the files for an oracle or engine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileConfig {
    pub version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serialized_type: Option<SerializedType>,
}

impl Default for FileConfig {
    fn default() -> Self {
        FileConfig {
            version: CURRENT_FILE_CONFIG_VERSION,
            serialized_type: None,
        }
    }
}

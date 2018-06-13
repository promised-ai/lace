pub mod csv;
pub mod generator;
pub mod sqlite;
pub mod traits;

pub use self::generator::StateBuilder;

use std::path::Path;

/// Denotes the source type of the data to be analyzed
pub enum DataSource {
    Sqlite(String),
    Postgres(String),
    Csv(String),
}

impl DataSource {
    pub fn to_path(&self) -> &Path {
        match &self {
            DataSource::Sqlite(s) => Path::new(s),
            DataSource::Postgres(s) => Path::new(s),
            DataSource::Csv(s) => Path::new(s),
        }
    }
}

/// Denotes the fiel type of the serialized `cc::State`s
pub enum SerializedType {
    Yaml,
    MessagePack,
    Json,
}

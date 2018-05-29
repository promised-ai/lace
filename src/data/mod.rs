pub mod csv;
pub mod sqlite;
pub mod traits;
pub mod generator;

pub use self::generator::StateBuilder;

/// Denotes the source type of the data to be analyzed
pub enum DataSource {
    Sqlite,
    Postgres,
    Csv,
}

/// Denotes the fiel type of the serialized `cc::State`s
pub enum SerializedType {
    Yaml,
    MessagePack,
    Json,
}

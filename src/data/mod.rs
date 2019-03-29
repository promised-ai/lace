pub mod csv;
pub mod data_source;
pub mod generator;
pub mod sqlite;
pub mod traits;

pub use self::data_source::DataSource;
pub use self::generator::StateBuilder;

/// Denotes the fiel type of the serialized `cc::State`s
pub enum SerializedType {
    Yaml,
    MessagePack,
    Json,
}

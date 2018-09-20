pub mod bencher;
pub mod engine;
pub mod engine_builder;
pub mod oracle;
pub mod utils;

use cc::DType;

pub type Given = Option<Vec<(usize, DType)>>;

pub use interface::bencher::Bencher;
pub use interface::engine::Engine;
pub use interface::engine_builder::EngineBuilder;
pub use interface::oracle::MiType;
pub use interface::oracle::Oracle;

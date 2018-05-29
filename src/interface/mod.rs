pub mod engine;
pub mod error;
pub mod oracle;
pub mod utils;
pub mod bencher;

use cc::DType;

pub type Given = Option<Vec<(usize, DType)>>;

pub use interface::engine::Engine;
pub use interface::bencher::Bencher;
pub use interface::oracle::MiType;
pub use interface::oracle::Oracle;

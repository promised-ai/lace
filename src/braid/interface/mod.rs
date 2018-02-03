pub mod oracle;
pub mod server;
pub mod error;
pub mod utils;

use cc::DType;

pub type Given = Option<Vec<(usize, DType)>>;

pub use interface::oracle::Oracle;
pub use interface::oracle::MiType;

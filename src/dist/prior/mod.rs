extern crate rand;
extern crate rv;

pub mod csd;
pub mod ng;

use self::rand::Rng;
pub use cc::ConjugateComponent;
pub use dist::prior::csd::Csd;
pub use dist::prior::ng::Ng;

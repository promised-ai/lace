pub mod traits;
pub mod bernoulli;
pub mod categorical;
pub mod gaussian;
pub mod prior;
pub mod suffstats;

pub use dist::gaussian::Gaussian;
pub use dist::bernoulli::Bernoulli;
pub use dist::categorical::Categorical;

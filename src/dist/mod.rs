pub mod traits;
pub mod bernoulli;
pub mod categorical;
pub mod dirichlet;
pub mod gaussian;
pub mod prior;


pub use dist::bernoulli::Bernoulli;
pub use dist::categorical::Categorical;
pub use dist::dirichlet::Dirichlet;
pub use dist::dirichlet::SymmetricDirichlet;
pub use dist::gaussian::Gaussian;

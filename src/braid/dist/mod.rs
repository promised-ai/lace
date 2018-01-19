pub mod traits;
pub mod bernoulli;
pub mod categorical;
pub mod dirichlet;
pub mod gaussian;
pub mod gamma;
pub mod prior;
pub mod mixture;


pub use dist::bernoulli::Bernoulli;
pub use dist::categorical::Categorical;
pub use dist::dirichlet::Dirichlet;
pub use dist::dirichlet::SymmetricDirichlet;
pub use dist::gaussian::Gaussian;
pub use dist::gamma::Gamma;
pub use dist::mixture::MixtureModel;

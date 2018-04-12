pub mod bernoulli;
pub mod categorical;
pub mod dirichlet;
pub mod gamma;
pub mod gaussian;
pub mod invgamma;
pub mod mixture;
pub mod prior;
pub mod traits;

pub use dist::bernoulli::Bernoulli;
pub use dist::categorical::Categorical;
pub use dist::dirichlet::Dirichlet;
pub use dist::dirichlet::SymmetricDirichlet;
pub use dist::gamma::Gamma;
pub use dist::gaussian::Gaussian;
pub use dist::invgamma::InvGamma;
pub use dist::mixture::MixtureModel;

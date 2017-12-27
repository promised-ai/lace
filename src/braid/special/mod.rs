pub mod erf;
pub mod beta;
pub mod gamma;
pub mod digamma;

pub use special::erf::erf;
pub use special::erf::erfinv;
pub use special::beta::beta;
pub use special::beta::betaln;
pub use special::gamma::gammaln_sign;
pub use special::gamma::gamma;
pub use special::gamma::gammaln;
pub use special::digamma::digamma;

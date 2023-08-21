use std::convert::TryFrom;

use lace_stats::rv::dist::{Bernoulli, Categorical, Gaussian, Poisson};
#[cfg(feature = "experimental")]
use lace_stats::rv::experimental::Sbd;

#[derive(Clone, Debug)]
/// A column mixture component
pub enum Component {
    /// Binary, Bernoulli component
    Binary(Bernoulli),
    /// Continuous, Gaussian component
    Continuous(Gaussian),
    /// Categorical, Discrete-Dirichlet component
    Categorical(Categorical),
    /// Count/Poisson component
    Count(Poisson),
    /// Index-type component
    #[cfg(feature = "experimental")]
    Index(Sbd),
}

macro_rules! impl_from_traits {
    ($inner: ty, $variant: ident) => {
        impl From<$inner> for Component {
            fn from(inner: $inner) -> Self {
                Component::$variant(inner)
            }
        }

        impl TryFrom<Component> for $inner {
            type Error = String;
            fn try_from(cpnt: Component) -> Result<Self, Self::Error> {
                match cpnt {
                    Component::$variant(inner) => Ok(inner),
                    _ => Err(String::from("Cannot convert Component")),
                }
            }
        }
    };
    ($inner: ident) => {
        impl_from_traits!($inner, $inner);
    };
}

impl_from_traits!(Gaussian, Continuous);
impl_from_traits!(Poisson, Count);
impl_from_traits!(Bernoulli, Binary);
impl_from_traits!(Categorical);
#[cfg(feature = "experimental")]
impl_from_traits!(Sbd, Index);

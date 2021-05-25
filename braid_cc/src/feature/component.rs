// FIXME: use try_from since from should never fail
use std::convert::From;

use braid_stats::labeler::Labeler;
use rv::dist::{Categorical, Gaussian, Poisson};

#[derive(Clone, Debug)]
/// A column mixture component
pub enum Component {
    /// Continuous, Gaussian component
    Continuous(Gaussian),
    /// Categorical, Discrete-Dirichlet component
    Categorical(Categorical),
    /// Labeler component
    Labeler(Labeler),
    /// Count/Poisson component
    Count(Poisson),
}

macro_rules! impl_from_traits {
    ($inner: ty, $variant: ident) => {
        impl From<$inner> for Component {
            fn from(inner: $inner) -> Self {
                Component::$variant(inner)
            }
        }

        impl From<Component> for $inner {
            fn from(cpnt: Component) -> Self {
                match cpnt {
                    Component::$variant(inner) => inner,
                    _ => panic!("Cannot convert Component"),
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
impl_from_traits!(Categorical);
impl_from_traits!(Labeler);

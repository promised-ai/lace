use std::convert::From;

use braid_stats::labeler::Labeler;
use rv::dist::{Categorical, Gaussian};

#[derive(Clone, Debug)]
pub enum Component {
    Continuous(Gaussian),
    Categorical(Categorical),
    Labeler(Labeler),
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
impl_from_traits!(Categorical);
impl_from_traits!(Labeler);

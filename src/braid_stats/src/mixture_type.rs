extern crate rv;

use self::rv::dist::{Categorical, Gaussian, Mixture};

pub enum MixtureType {
    Gaussian(Mixture<Gaussian>),
    Categorical(Mixture<Categorical>),
}

impl MixtureType {
    pub fn is_gaussian(&self) -> bool {
        match self {
            MixtureType::Gaussian(..) => true,
            _ => false,
        }
    }

    pub fn is_categorial(&self) -> bool {
        match self {
            MixtureType::Categorical(..) => true,
            _ => false,
        }
    }

    pub fn unwrap_gaussian(self) -> Mixture<Gaussian> {
        match self {
            MixtureType::Gaussian(m) => m,
            _ => panic!("not a Gaussian Mixture"),
        }
    }

    pub fn unwrap_categorical(self) -> Mixture<Categorical> {
        match self {
            MixtureType::Categorical(m) => m,
            _ => panic!("not a Categorical Mixture"),
        }
    }
}

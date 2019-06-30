use std::convert::From;

use rv::dist::{Categorical, Gaussian, Mixture};
use rv::traits::{Entropy, Rv};

use crate::labeler::Labeler;
use crate::MixtureJsd;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum MixtureType {
    Gaussian(Mixture<Gaussian>),
    Categorical(Mixture<Categorical>),
    Labeler(Mixture<Labeler>),
}

macro_rules! mt_combine_arm {
    ($variant: ident, $mixtures: ident) => {{
        let mixtures: Vec<Mixture<$variant>> = $mixtures
            .drain(..)
            .map(|mt| match mt {
                MixtureType::$variant(inner) => inner,
                _ => panic!("Cannot combine different MixtureType variants"),
            })
            .collect();
        let combined = Mixture::combine(mixtures).unwrap();
        MixtureType::$variant(combined)
    }};
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

    pub fn is_labeler(&self) -> bool {
        match self {
            MixtureType::Labeler(..) => true,
            _ => false,
        }
    }

    /// Get the numbre of componentns in this mixture
    pub fn k(&self) -> usize {
        match self {
            MixtureType::Categorical(mm) => mm.k(),
            MixtureType::Gaussian(mm) => mm.k(),
            MixtureType::Labeler(mm) => mm.k(),
        }
    }

    /// Combine MixtureTypes into a MixtureType containing all components. Will
    /// panic if MixtureType variants do not match.
    pub fn combine(mut mixtures: Vec<MixtureType>) -> MixtureType {
        match mixtures[0] {
            MixtureType::Categorical(..) => {
                mt_combine_arm!(Categorical, mixtures)
            }
            MixtureType::Gaussian(..) => mt_combine_arm!(Gaussian, mixtures),
            MixtureType::Labeler(..) => mt_combine_arm!(Labeler, mixtures),
        }
    }
}

macro_rules! impl_from {
    ($fx: ident) => {
        impl From<Mixture<$fx>> for MixtureType {
            fn from(mm: Mixture<$fx>) -> MixtureType {
                MixtureType::$fx(mm)
            }
        }

        impl From<MixtureType> for Mixture<$fx> {
            fn from(mt: MixtureType) -> Mixture<$fx> {
                match mt {
                    MixtureType::$fx(mm) => mm,
                    _ => panic!("Invalid inner type for conversion"),
                }
            }
        }
    };
}

impl Entropy for MixtureType {
    fn entropy(&self) -> f64 {
        match self {
            MixtureType::Gaussian(mm) => mm.entropy(),
            MixtureType::Categorical(mm) => mm.entropy(),
            MixtureType::Labeler(mm) => {
                super::labeler::ALL_LABELS.iter().fold(0.0, |acc, x| {
                    let p = mm.f(&x);
                    acc - p * p.ln()
                })
            }
        }
    }
}

impl_from!(Gaussian);
impl_from!(Categorical);
impl_from!(Labeler);

impl<Fx> MixtureJsd for Mixture<Fx>
where
    Fx: Entropy,
    Mixture<Fx>: Entropy,
{
    fn mixture_jsd(&self) -> f64 {
        let h_mixture = self.entropy();
        let h_components = self
            .weights
            .iter()
            .zip(self.components.iter())
            .fold(0_f64, |acc, (w, cpnt)| acc + w * cpnt.entropy());
        h_mixture - h_components
    }
}

impl MixtureJsd for MixtureType {
    fn mixture_jsd(&self) -> f64 {
        match self {
            MixtureType::Gaussian(mm) => mm.mixture_jsd(),
            MixtureType::Categorical(mm) => mm.mixture_jsd(),
            MixtureType::Labeler(mm) => {
                let h_mixture = self.entropy();
                let h_components = mm
                    .weights
                    .iter()
                    .zip(mm.components.iter())
                    .fold(0_f64, |acc, (w, cpnt)| acc + w * cpnt.entropy());
                h_mixture - h_components
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k() {
        let mm = Mixture::uniform(vec![Gaussian::default(); 5]).unwrap();
        let mt = MixtureType::Gaussian(mm);
        assert_eq!(mt.k(), 5);
    }

    #[test]
    fn combine() {
        let mts: Vec<MixtureType> = (1..=5)
            .map(|k| {
                let mm =
                    Mixture::uniform(vec![Gaussian::default(); k]).unwrap();
                MixtureType::Gaussian(mm)
            })
            .collect();
        let combined = MixtureType::combine(mts);
        assert_eq!(combined.k(), 1 + 2 + 3 + 4 + 5);
    }
}

use std::convert::From;

use crate::rv::dist::{Bernoulli, Categorical, Gaussian, Mixture, Poisson};
use crate::rv::experimental::stick_breaking::StickBreakingDiscrete;
use crate::rv::traits::Entropy;
use crate::MixtureJsd;

/// Enum describing the types of mixture models that can be constructed from
/// Lace column models.
#[derive(Clone, Debug, PartialEq)]
pub enum MixtureType {
    Bernoulli(Mixture<Bernoulli>),
    Gaussian(Mixture<Gaussian>),
    Categorical(Mixture<Categorical>),
    Poisson(Mixture<Poisson>),
    StickBreakingDiscrete(Mixture<StickBreakingDiscrete>),
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
        let combined = Mixture::combine(mixtures);
        MixtureType::$variant(combined)
    }};
}

impl MixtureType {
    /// Returns `True` if the mixture is Bernoulli
    pub fn is_bernoulli(&self) -> bool {
        matches!(self, MixtureType::Bernoulli(..))
    }

    /// Returns `True` if the mixture is Gaussian
    pub fn is_gaussian(&self) -> bool {
        matches!(self, MixtureType::Gaussian(..))
    }

    /// Returns `True` if the mixture is Categorical
    pub fn is_categorial(&self) -> bool {
        matches!(self, MixtureType::Categorical(..))
    }

    /// Returns `True` if the mixture is Poisson
    pub fn is_poisson(&self) -> bool {
        matches!(self, MixtureType::Poisson(..))
    }

    /// Get the number of components in this mixture
    pub fn k(&self) -> usize {
        match self {
            MixtureType::Bernoulli(mm) => mm.k(),
            MixtureType::Categorical(mm) => mm.k(),
            MixtureType::Gaussian(mm) => mm.k(),
            MixtureType::Poisson(mm) => mm.k(),
            MixtureType::StickBreakingDiscrete(mm) => mm.k(),
        }
    }

    /// Combine MixtureTypes into a MixtureType containing all components. Will
    /// panic if MixtureType variants do not match.
    pub fn combine(mut mixtures: Vec<MixtureType>) -> MixtureType {
        match mixtures[0] {
            MixtureType::Bernoulli(..) => {
                // TODO: can optimize  by combining into a one component
                // bernoulli that is the mean of all the components
                mt_combine_arm!(Bernoulli, mixtures)
            }
            MixtureType::Categorical(..) => {
                mt_combine_arm!(Categorical, mixtures)
            }
            MixtureType::Gaussian(..) => mt_combine_arm!(Gaussian, mixtures),
            MixtureType::Poisson(..) => mt_combine_arm!(Poisson, mixtures),
            MixtureType::StickBreakingDiscrete(..) => {
                mt_combine_arm!(StickBreakingDiscrete, mixtures)
            }
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

        // TODO: should be try from
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
            MixtureType::Bernoulli(mm) => mm.entropy(),
            MixtureType::Gaussian(mm) => mm.entropy(),
            MixtureType::Categorical(mm) => mm.entropy(),
            MixtureType::Poisson(mm) => mm.entropy(),
            MixtureType::StickBreakingDiscrete(mm) => {
                // FIXME: impl in rv!
                // mm.entropy(),
                0.0
            }
        }
    }
}

impl_from!(Gaussian);
impl_from!(Categorical);
impl_from!(Poisson);
impl_from!(Bernoulli);
impl_from!(StickBreakingDiscrete);

impl<Fx> MixtureJsd for Mixture<Fx>
where
    Fx: Entropy,
    Mixture<Fx>: Entropy,
{
    fn mixture_jsd(&self) -> f64 {
        let h_mixture = self.entropy();
        let h_components = self
            .weights()
            .iter()
            .zip(self.components().iter())
            .fold(0_f64, |acc, (w, cpnt)| acc + w * cpnt.entropy());
        h_mixture - h_components
    }
}

impl MixtureJsd for MixtureType {
    fn mixture_jsd(&self) -> f64 {
        match self {
            MixtureType::Bernoulli(mm) => mm.mixture_jsd(),
            MixtureType::Gaussian(mm) => mm.mixture_jsd(),
            MixtureType::Categorical(mm) => mm.mixture_jsd(),
            MixtureType::Poisson(mm) => mm.mixture_jsd(),
            MixtureType::StickBreakingDiscrete(mm) => {
                // FIXME: impl in rv!
                // mm.mixture_jsd(),
                0.0
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

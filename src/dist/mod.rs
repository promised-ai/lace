//! Probability distribution utilities and traits
pub mod stick_breaking;
pub mod traits;

use std::convert::TryFrom;
use std::fmt::Debug;

use braid_stats::{Datum, UpdatePrior};
use rv::traits::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::cc::Component;
use crate::dist::traits::AccumScore;

/// A Braid-ready datum.
pub trait BraidDatum:
    Sync + Serialize + DeserializeOwned + TryFrom<Datum> + Default + Clone + Debug
{
}

impl<X> BraidDatum for X where
    X: Sync
        + Serialize
        + DeserializeOwned
        + TryFrom<Datum>
        + Default
        + Clone
        + Debug
{
}

/// A Braid-ready datum.
pub trait BraidStat:
    Sync + Serialize + DeserializeOwned + Debug + Clone + PartialEq
{
}
impl<X> BraidStat for X where
    X: Sync + Serialize + DeserializeOwned + Debug + Clone + PartialEq
{
}

/// A Braid-ready likelihood function, f(x).
pub trait BraidLikelihood<X: BraidDatum>:
    Rv<X>
    + AccumScore<X>
    + HasSuffStat<X>
    + Serialize
    + DeserializeOwned
    + Sync
    + Into<Component>
    + Clone
    + Debug
    + PartialEq
{
}

impl<X, Fx> BraidLikelihood<X> for Fx
where
    X: BraidDatum,
    Fx: Rv<X>
        + AccumScore<X>
        + HasSuffStat<X>
        + Serialize
        + DeserializeOwned
        + Sync
        + Into<Component>
        + Clone
        + Debug
        + PartialEq,
    Fx::Stat: Sync + Serialize + DeserializeOwned + Clone + Debug,
{
}

/// A Braid-ready prior π(f)
pub trait BraidPrior<X: BraidDatum, Fx: BraidLikelihood<X>>:
    ConjugatePrior<X, Fx>
    + UpdatePrior<X, Fx>
    + Serialize
    + DeserializeOwned
    + Sync
    + Clone
    + Debug
{
    fn empty_suffstat(&self) -> Fx::Stat;
}

use braid_stats::prior::Csd;
use rv::data::CategoricalSuffStat;
use rv::dist::Categorical;

impl BraidPrior<u8, Categorical> for Csd {
    fn empty_suffstat(&self) -> CategoricalSuffStat {
        CategoricalSuffStat::new(self.symdir.k())
    }
}

use braid_stats::prior::Pg;
use rv::data::PoissonSuffStat;
use rv::dist::Poisson;

impl BraidPrior<u32, Poisson> for Pg {
    fn empty_suffstat(&self) -> PoissonSuffStat {
        PoissonSuffStat::new()
    }
}

use braid_stats::prior::Ng;
use rv::data::GaussianSuffStat;
use rv::dist::Gaussian;

impl BraidPrior<f64, Gaussian> for Ng {
    fn empty_suffstat(&self) -> GaussianSuffStat {
        GaussianSuffStat::new()
    }
}

use braid_stats::labeler::Label;
use braid_stats::labeler::Labeler;
use braid_stats::labeler::LabelerPrior;
use braid_stats::labeler::LabelerSuffStat;

impl BraidPrior<Label, Labeler> for LabelerPrior {
    fn empty_suffstat(&self) -> LabelerSuffStat {
        LabelerSuffStat::new()
    }
}

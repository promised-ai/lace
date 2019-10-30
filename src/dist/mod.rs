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
    Sync + Serialize + DeserializeOwned + Debug + Clone
{
}
impl<X> BraidStat for X where
    X: Sync + Serialize + DeserializeOwned + Debug + Clone
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
        + Debug,
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
}

impl<X, Fx, Pr> BraidPrior<X, Fx> for Pr
where
    Pr: ConjugatePrior<X, Fx>
        + UpdatePrior<X, Fx>
        + Serialize
        + DeserializeOwned
        + Sync
        + Clone
        + Debug,
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
{
}

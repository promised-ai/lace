pub mod stick_breaking;
pub mod traits;

use std::convert::TryFrom;
use std::fmt::Debug;

use braid_stats::UpdatePrior;
use rv::traits::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::cc::Datum;
use crate::dist::traits::AccumScore;

/// A Braid-ready datum.
pub trait BraidDatum:
    Debug + Sync + Clone + Serialize + DeserializeOwned + TryFrom<Datum> + Default
{
}
impl<X> BraidDatum for X where
    X: Debug
        + Sync
        + Clone
        + Serialize
        + DeserializeOwned
        + TryFrom<Datum>
        + Default
{
}

/// A Braid-ready datum.
pub trait BraidStat:
    Debug + Sync + Clone + Serialize + DeserializeOwned
{
}
impl<X> BraidStat for X where
    X: Debug + Sync + Clone + Serialize + DeserializeOwned
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
        + Clone
        + Debug,
    Fx::Stat: Sync + Serialize + DeserializeOwned + Clone,
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

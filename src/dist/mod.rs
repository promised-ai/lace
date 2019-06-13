pub mod stick_breaking;
pub mod traits;

use std::convert::TryFrom;

use braid_stats::UpdatePrior;
use rv::traits::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::cc::{Component, Datum};
use crate::dist::traits::AccumScore;

/// A Braid-ready datum.
pub trait BraidDatum:
    Sync + Serialize + DeserializeOwned + TryFrom<Datum> + Default + ApiReady
{
}

impl<X> BraidDatum for X where
    X: Sync
        + Serialize
        + DeserializeOwned
        + TryFrom<Datum>
        + Default
        + ApiReady
{
}

/// A Braid-ready datum.
pub trait BraidStat: Sync + Serialize + DeserializeOwned + ApiReady {}
impl<X> BraidStat for X where X: Sync + Serialize + DeserializeOwned + ApiReady {}

/// A Braid-ready likelihood function, f(x).
pub trait BraidLikelihood<X: BraidDatum>:
    Rv<X>
    + AccumScore<X>
    + HasSuffStat<X>
    + Serialize
    + DeserializeOwned
    + Sync
    + ApiReady
    + Into<Component>
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
        + ApiReady
        + Into<Component>,
    Fx::Stat: Sync + Serialize + DeserializeOwned + ApiReady,
{
}

/// A Braid-ready prior π(f)
pub trait BraidPrior<X: BraidDatum, Fx: BraidLikelihood<X>>:
    ConjugatePrior<X, Fx>
    + UpdatePrior<X, Fx>
    + Serialize
    + DeserializeOwned
    + Sync
    + ApiReady
{
}

impl<X, Fx, Pr> BraidPrior<X, Fx> for Pr
where
    Pr: ConjugatePrior<X, Fx>
        + UpdatePrior<X, Fx>
        + Serialize
        + DeserializeOwned
        + Sync
        + ApiReady,
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
{
}

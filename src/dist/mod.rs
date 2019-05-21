pub mod stick_breaking;
pub mod traits;

extern crate braid_stats;
extern crate rv;
extern crate serde;

use std::convert::TryFrom;
use std::fmt::Debug;

use self::braid_stats::UpdatePrior;
use self::rv::traits::*;
use self::serde::de::DeserializeOwned;
use self::serde::Serialize;

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

// pub trait UpdatePrior<X, Fx: Rv<X>> {
//     /// Draw new prior parameters given a set of existing models and the hyper
//     /// prior.
//     fn update_prior<R: Rng>(&mut self, components: &Vec<&Fx>, rng: &mut R);
// }

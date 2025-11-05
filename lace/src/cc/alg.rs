//! Data types for choosing different methods of sampling in CrossCat
//!
//! There are currently four algorithms that have their own special
//! considerations.
//!
//!
//! ## Gibbs
//!
//! This is the collapsed Gibbs sampler outlined by Radford Neal. In this method
//! rows/columns are randomly removed and then reinserted into the table
//! according to the contribution to the partition and their likelihood under
//! each component.
//!
//! ### Pros
//!
//! - Simple
//! - Valid MCMC
//!
//! ### Cons
//!
//! - Slow due to random access and moderately expensive posterior predictive
//!   computation
//! - Mixes slowly due to moving only one row/column at a time
//!
//! ### Citations
//!
//! Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process
//!     mixture models. Journal of computational and graphical statistics, 9(2),
//!     249-265.
//!
//!
//! ## Sequential Adaptive Merge-Split (Sams)
//!
//! Two rows/columns are selected at random. If they are in the same component,
//! a split is proposed using an adaptive restricted Gibbs sweep; if they are in
//! different components, a merge of the two components is proposed.
//!
//! ### Pros
//! - Fast since only subsets of the data are looked at
//! - Potential for big moves since Sams operates on swaths of data
//!
//! ### Cons
//! - Many proposals are rejected, especially merge proposals
//! - Difficult to do fine-grained moves. It's best to pair with another
//!   algorithms.
//!
//! ### Citations
//!
//! Jain, S., & Neal, R. M. (2004). A split-merge Markov chain Monte Carlo
//!     procedure for the Dirichlet process mixture model. Journal of
//!     computational and Graphical Statistics, 13(1), 158-182.
//!
//!
//! ## Finite Approximation (FiniteCpu)
//!
//! In this algorithm a new component is proposed from the prior, then each
//! row/column is reassigned according to its likelihood under each component.
//! This algorithm does not marginalize away the component parameters like
//! `Sams` and `Gibbs`, but explicitly represent the component parameters.
//!
//! ### Pros
//! - Extremely fast due to cache efficiency and parallelization
//! - Simple
//!
//! ### Cons
//! - Not a valid MCMC method. It only approximates the distribution so it tends
//!   to create fewer components than there should be, which can cause
//!   under-fitting.
//! - Slow to mix since rows/columns are moved individually and since the new
//!   component is drawn from the prior, it is never guaranteed to fit to
//!   anything particularly well.
//!
//!
//! ## Slice sampler (Slice)
//!
//! This is nearly identical to FiniteCpu, but uses a stick-breaking prior and a
//! slice (or beam) variable to approximate the correct distribution.
//!
//! **NOTE:** It is best to use a Gamma prior (not InvGamma) on the CRP α when
//! using this algorithm.
//!
//! ### Pros
//! - Fast
//! - Correct MCMC
//!
//! ### Cons
//! - Users must be careful when using a prior on CRP α that has high or
//!   infinite variance. If α is too high, the stick will break indefinitely and
//!   cause an panic.
use std::fmt;
use std::str::FromStr;

use serde::Deserialize;
use serde::Serialize;

use crate::ParseError;

/// The MCMC algorithm to use for row reassignment
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    #[serde(rename = "finite_cpu")]
    FiniteCpu,
    /// An Improved slice sampler based on stick breaking:
    ///
    /// Ge, H., Chen, Y., Wan, M., & Ghahramani, Z. (2015, June). Distributed
    ///    inference for Dirichlet process mixture models. In International
    ///    Conference on Machine Learning (pp. 2276-2284).
    #[serde(rename = "slice")]
    Slice,
    /// Sequential importance sampling split-merge
    #[serde(rename = "sams")]
    Sams,
    /// Sequential, enumerative Gibbs
    #[serde(rename = "gibbs")]
    Gibbs,
}

impl fmt::Display for RowAssignAlg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            RowAssignAlg::FiniteCpu => "FiniteCpu",
            RowAssignAlg::Gibbs => "Gibbs",
            RowAssignAlg::Slice => "Slice",
            RowAssignAlg::Sams => "Sams",
        };
        write!(f, "{s}")
    }
}

// implemented so we can use as CLI args
impl FromStr for RowAssignAlg {
    type Err = ParseError<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "finite_cpu" => Ok(RowAssignAlg::FiniteCpu),
            "gibbs" => Ok(RowAssignAlg::Gibbs),
            "slice" => Ok(RowAssignAlg::Slice),
            "sams" => Ok(RowAssignAlg::Sams),
            _ => Err(ParseError(s.to_owned())),
        }
    }
}

impl<'s> From<&'s str> for RowAssignAlg {
    fn from(value: &'s str) -> Self {
        Self::from_str(value).unwrap()
    }
}

/// The MCMC algorithm to use for column reassignment
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub enum ColAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    #[serde(rename = "finite_cpu")]
    FiniteCpu,
    /// Sequential, enumerative Gibbs
    #[serde(rename = "gibbs")]
    Gibbs,
    /// An Improved slice sampler based on stick breaking:
    ///
    /// Ge, H., Chen, Y., Wan, M., & Ghahramani, Z. (2015, June). Distributed
    ///    inference for Dirichlet process mixture models. In International
    ///    Conference on Machine Learning (pp. 2276-2284).
    #[serde(rename = "slice")]
    Slice,
}

impl fmt::Display for ColAssignAlg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            ColAssignAlg::FiniteCpu => "FiniteCpu",
            ColAssignAlg::Gibbs => "Gibbs",
            ColAssignAlg::Slice => "Slice",
        };
        write!(f, "{s}")
    }
}

impl FromStr for ColAssignAlg {
    type Err = ParseError<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "finite_cpu" => Ok(ColAssignAlg::FiniteCpu),
            "gibbs" => Ok(ColAssignAlg::Gibbs),
            "slice" => Ok(ColAssignAlg::Slice),
            _ => Err(ParseError(s.to_owned())),
        }
    }
}

impl<'s> From<&'s str> for ColAssignAlg {
    fn from(value: &'s str) -> Self {
        Self::from_str(value).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_alg_from_string_finite_cpu() {
        assert_eq!(
            RowAssignAlg::from_str("finite_cpu").unwrap(),
            RowAssignAlg::FiniteCpu
        );
    }

    #[test]
    fn test_row_alg_from_string_slice() {
        assert_eq!(
            RowAssignAlg::from_str("slice").unwrap(),
            RowAssignAlg::Slice
        );
    }

    #[test]
    fn test_row_alg_from_string_gibbs() {
        assert_eq!(
            RowAssignAlg::from_str("gibbs").unwrap(),
            RowAssignAlg::Gibbs
        );
    }

    #[test]
    fn test_row_alg_from_string_sams() {
        assert_eq!(RowAssignAlg::from_str("sams").unwrap(), RowAssignAlg::Sams);
    }

    #[test]
    fn test_row_alg_from_string_invalid() {
        assert!(RowAssignAlg::from_str("finte").is_err());
    }

    #[test]
    fn test_col_alg_from_string_finite_cpu() {
        assert_eq!(
            ColAssignAlg::from_str("finite_cpu").unwrap(),
            ColAssignAlg::FiniteCpu
        );
    }

    #[test]
    fn test_col_alg_from_string_slice() {
        assert_eq!(
            ColAssignAlg::from_str("slice").unwrap(),
            ColAssignAlg::Slice
        );
    }

    #[test]
    fn test_col_alg_from_string_gibbs() {
        assert_eq!(
            ColAssignAlg::from_str("gibbs").unwrap(),
            ColAssignAlg::Gibbs
        );
    }

    #[test]
    fn test_col_alg_from_string_invalid() {
        assert!(ColAssignAlg::from_str("finte").is_err());
    }
}

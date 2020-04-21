//! Data types for choosing different methods of sampling crosscat
use crate::ParseError;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

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
        write!(f, "{}", s)
    }
}

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
        write!(f, "{}", s)
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

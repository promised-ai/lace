//! Data types for choosing different methods of sampling crosscat
use crate::result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// The MCMC algorithm to use for row reassignment
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq, Hash)]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    #[serde(rename = "finite_cpu")]
    FiniteCpu,
    /// OpenCL GPU-parallelized finite Dirichlet approximation
    #[serde(rename = "finite_gpu")]
    FiniteGpu,
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
            RowAssignAlg::FiniteGpu => "FiniteGpu",
            RowAssignAlg::Gibbs => "Gibbs",
            RowAssignAlg::Slice => "Slice",
            RowAssignAlg::Sams => "Sams",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for RowAssignAlg {
    type Err = result::Error;

    fn from_str(s: &str) -> result::Result<Self> {
        match s {
            "finite_cpu" => Ok(RowAssignAlg::FiniteCpu),
            "finite_gpu" => Ok(RowAssignAlg::FiniteGpu),
            "gibbs" => Ok(RowAssignAlg::Gibbs),
            "slice" => Ok(RowAssignAlg::Slice),
            "sams" => Ok(RowAssignAlg::Sams),
            _ => {
                let err_kind = result::ErrorKind::ParseError;
                let msg =
                    format!("Could not parse row assignment algorithm '{}'", s);
                Err(result::Error::new(err_kind, msg))
            }
        }
    }
}

/// The MCMC algorithm to use for column reassignment
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq, Hash)]
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
    type Err = result::Error;

    fn from_str(s: &str) -> result::Result<Self> {
        match s {
            "finite_cpu" => Ok(ColAssignAlg::FiniteCpu),
            "gibbs" => Ok(ColAssignAlg::Gibbs),
            "slice" => Ok(ColAssignAlg::Slice),
            _ => {
                let err_kind = result::ErrorKind::ParseError;
                let msg = format!(
                    "Could not parse column assignment algorithm '{}'",
                    s
                );
                Err(result::Error::new(err_kind, msg))
            }
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
    fn test_row_alg_from_string_finite_gpu() {
        assert_eq!(
            RowAssignAlg::from_str("finite_gpu").unwrap(),
            RowAssignAlg::FiniteGpu
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

use std::fmt;

/// The MCMC algorithm to use for row reassignment
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet appproximation
    #[serde(rename = "finite_cpu")]
    FiniteCpu,
    /// OpenCL GPU-parallelized finite Dirichlet appproximation
    #[serde(rename = "finite_gpu")]
    FiniteGpu,
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
            RowAssignAlg::Sams => "Samms",
        };
        write!(f, "{}", s)
    }
}

/// The MCMC algorithm to use for column reassignment
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum ColAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    #[serde(rename = "finite_cpu")]
    FiniteCpu,
    /// Sequential, enumerative Gibbs
    #[serde(rename = "gibbs")]
    Gibbs,
}

impl fmt::Display for ColAssignAlg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            ColAssignAlg::FiniteCpu => "FiniteCpu",
            ColAssignAlg::Gibbs => "Gibbs",
        };
        write!(f, "{}", s)
    }
}

pub const DEFAULT_ROW_ASSIGN_ALG: RowAssignAlg = RowAssignAlg::FiniteCpu;
pub const DEFAULT_COL_ASSIGN_ALG: ColAssignAlg = ColAssignAlg::FiniteCpu;

/// The MCMC algorithm to use for row reassignment
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet appproximation
    #[serde(rename = "finite_cpu")]
    FiniteCpu,
    /// OpenCL GPU-parallelized finite Dirichlet appproximation
    #[serde(rename = "finite_gpu")]
    FiniteGpu,
    /// Sequential importance samplint split-merge
    #[serde(rename = "split_merge")]
    SplitMerge,
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

pub const DEFAULT_ROW_ASSIGN_ALG: RowAssignAlg = RowAssignAlg::FiniteCpu;
pub const DEFAULT_COL_ASSIGN_ALG: ColAssignAlg = ColAssignAlg::FiniteCpu;

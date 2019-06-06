//! Default values for priors and inference-type things
use crate::cc::{ColAssignAlg, RowAssignAlg};
use rv::dist::Gamma;

/// The default number of iterations of the mh_prior sampler
pub const MH_PRIOR_ITERS: usize = 50;

/// Default alpha prior for Geweke
pub const GEWEKE_ALPHA_PRIOR: Gamma = Gamma {
    shape: 3.0,
    rate: 3.0,
};

/// Default alpha prior in general
pub const GENERAL_ALPHA_PRIOR: Gamma = Gamma {
    shape: 1.0,
    rate: 1.0,
};

/// Default alpha prior for State assignment of columns to views
pub const STATE_ALPHA_PRIOR: Gamma = Gamma {
    shape: 1.0,
    rate: 1.0,
};

/// Default alpha prior for View assignment of rows to categories
pub const VIEW_ALPHA_PRIOR: Gamma = Gamma {
    shape: 1.0,
    rate: 1.0,
};

/// Deafult row re-assignment algorithm
pub const ROW_ASSIGN_ALG: RowAssignAlg = RowAssignAlg::FiniteCpu;

/// Deafult column re-assignment kernel algorithm
pub const COL_ASSIGN_ALG: ColAssignAlg = ColAssignAlg::FiniteCpu;

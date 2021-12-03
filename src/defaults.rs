//! Default values
use braid_cc::alg::{ColAssignAlg, RowAssignAlg};

/// Deafult row re-assignment algorithm
pub const ROW_ASSIGN_ALG: RowAssignAlg = RowAssignAlg::FiniteCpu;

/// Deafult column re-assignment kernel algorithm
pub const COL_ASSIGN_ALG: ColAssignAlg = ColAssignAlg::FiniteCpu;

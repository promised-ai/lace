//! Default values
use crate::cc::alg::{ColAssignAlg, RowAssignAlg};

/// Default row re-assignment algorithm
pub const ROW_ASSIGN_ALG: RowAssignAlg = RowAssignAlg::Slice;

/// Default column re-assignment kernel algorithm
pub const COL_ASSIGN_ALG: ColAssignAlg = ColAssignAlg::Slice;

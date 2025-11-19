//! Default values
use crate::cc::alg::ColAssignAlg;
use crate::cc::alg::RowAssignAlg;

/// Default row re-assignment algorithm
pub const ROW_ASSIGN_ALG: RowAssignAlg = RowAssignAlg::Slice;

/// Default column re-assignment kernel algorithm
pub const COL_ASSIGN_ALG: ColAssignAlg = ColAssignAlg::Slice;

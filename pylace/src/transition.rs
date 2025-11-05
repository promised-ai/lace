use lace::cc::alg::ColAssignAlg;
use lace::cc::alg::RowAssignAlg;
use pyo3::prelude::*;

/// A column reassignment MCMC kernel
#[pyclass]
#[derive(Clone, Copy)]
pub(crate) struct ColumnKernel(ColAssignAlg);

#[pymethods]
impl ColumnKernel {
    /// The `slice` column reassignment kernel
    #[staticmethod]
    fn slice() -> Self {
        Self(ColAssignAlg::Slice)
    }

    /// The `gibbs` column reassignment kernel
    #[staticmethod]
    fn gibbs() -> Self {
        Self(ColAssignAlg::Gibbs)
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// A row reassignment MCMC kernel
#[pyclass]
#[derive(Clone, Copy)]
pub(crate) struct RowKernel(RowAssignAlg);

#[pymethods]
impl RowKernel {
    #[staticmethod]
    /// The `slice` row reassignment kernel
    fn slice() -> Self {
        Self(RowAssignAlg::Slice)
    }

    #[staticmethod]
    /// The `gibbs` row reassignment kernel
    fn gibbs() -> Self {
        Self(RowAssignAlg::Gibbs)
    }

    #[staticmethod]
    /// The `sams` merge-split row reassignment kernel
    fn sams() -> Self {
        Self(RowAssignAlg::Sams)
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// A particular state transition within the Markov chain
#[pyclass]
#[derive(Clone, Copy)]
pub(crate) struct StateTransition(lace::cc::transition::StateTransition);

#[pymethods]
impl StateTransition {
    /// The column reassignment transition with selected MCMC kernel
    #[staticmethod]
    fn column_assignment(kernel: ColumnKernel) -> Self {
        Self(lace::cc::transition::StateTransition::ColumnAssignment(
            kernel.0,
        ))
    }

    /// The row reassignment transition with selected MCMC kernel
    #[staticmethod]
    fn row_assignment(kernel: RowKernel) -> Self {
        Self(lace::cc::transition::StateTransition::RowAssignment(
            kernel.0,
        ))
    }

    /// The state prior process parameters (controls the assignment of
    /// columns to views) transition.
    #[staticmethod]
    fn state_prior_process_params() -> Self {
        Self(lace::cc::transition::StateTransition::StatePriorProcessParams)
    }

    /// The view prior process parameters (controls the assignment of rows to
    /// categories within each view) transition.
    #[staticmethod]
    fn view_prior_process_params() -> Self {
        Self(lace::cc::transition::StateTransition::ViewPriorProcessParams)
    }

    /// Re-sample the feature prior parameters
    #[staticmethod]
    fn feature_priors() -> Self {
        Self(lace::cc::transition::StateTransition::FeaturePriors)
    }

    ///
    #[staticmethod]
    fn component_parameters() -> Self {
        Self(lace::cc::transition::StateTransition::ComponentParams)
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl From<StateTransition> for lace::cc::transition::StateTransition {
    fn from(val: StateTransition) -> Self {
        val.0
    }
}

impl std::fmt::Display for RowKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            RowAssignAlg::Slice => write!(f, "RowKernel::Slice"),
            RowAssignAlg::Gibbs => write!(f, "RowKernel::Gibbs"),
            RowAssignAlg::Sams => write!(f, "RowKernel::Sams"),
            RowAssignAlg::FiniteCpu => write!(f, "RowKernel::Finite"),
        }
    }
}

impl std::fmt::Display for ColumnKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ColAssignAlg::Slice => write!(f, "ColumnKernel::Slice"),
            ColAssignAlg::Gibbs => write!(f, "ColumnKernel::Gibbs"),
            ColAssignAlg::FiniteCpu => write!(f, "ColumnKernel::Finite"),
        }
    }
}

impl std::fmt::Display for StateTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use lace::cc::transition::StateTransition as St;
        match self.0 {
            St::ColumnAssignment(kernel) => {
                write!(f, "ColumnAssignment({kernel})")
            }
            St::RowAssignment(kernel) => {
                write!(f, "ColumnAssignment({kernel})")
            }
            St::FeaturePriors => write!(f, "FeaturePriors"),
            St::ComponentParams => write!(f, "ComponentParams"),
            St::StatePriorProcessParams => write!(f, "StatePriorProcessParams"),
            St::ViewPriorProcessParams => write!(f, "ViewPriorProcessParams"),
        }
    }
}

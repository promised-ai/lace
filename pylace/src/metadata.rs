use lace::codebook::{ColMetadata, ColType};
use lace::stats::prior::csd::CsdHyper;
use lace::stats::prior::nix::NixHyper;
use lace::stats::rv::dist::{
    Gamma, Gaussian, InvGamma, NormalInvChiSquared, SymmetricDirichlet,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::utils::to_pyerr;

/// Column metadata
#[pyclass]
#[derive(Clone, Debug)]
pub struct ColumnMetadata(pub ColMetadata);

/// Prior on continuous column data
#[pyclass]
#[derive(Clone, Debug)]
pub struct ContinuousPrior(pub NormalInvChiSquared);

/// Hyperprior on continuous column prior
#[pyclass]
#[derive(Clone, Debug)]
pub struct ContinuousHyper(pub NixHyper);

#[pymethods]
impl ContinuousPrior {
    /// Create a new ``ContinuousPrior``
    ///
    /// Parameters
    /// ----------
    /// m: float
    ///     The prior mean
    /// k: float
    /// v: float
    /// s2: float
    #[new]
    #[pyo3(signature = (m=0.0, k=1.0, v=1.0, s2=1.0))]
    pub fn new(m: f64, k: f64, v: f64, s2: f64) -> PyResult<Self> {
        let inner = NormalInvChiSquared::new(m, k, v, s2).map_err(to_pyerr)?;
        Ok(Self(inner))
    }
}

#[pymethods]
impl ContinuousHyper {
    /// Create a new ``ContinuousHyper``
    ///
    /// Parameters
    /// ----------
    /// pr_m: (float, float)
    ///     The mean and standard deviation of the normal distribution on the
    ///     prior mean.
    /// pr_k: (float, float)
    ///     The shape and rate parameters of the gamma distribution on ``k``
    /// pr_v: (float, float)
    ///     The shape and scale parameters of the inverse gamma distribution on
    ///     ``v``.
    /// pr_s2: (float, float)
    ///     The shape and scale parameters of the inverse gamma distribution on
    ///     ``s2``.
    #[new]
    #[pyo3(signature = (pr_m=(0.0, 1.0), pr_k=(1.0, 1.0), pr_v=(2.0, 2.0), pr_s2=(2.0, 2.0)))]
    pub fn new(
        pr_m: (f64, f64),
        pr_k: (f64, f64),
        pr_v: (f64, f64),
        pr_s2: (f64, f64),
    ) -> PyResult<Self> {
        Ok(Self(NixHyper {
            pr_m: Gaussian::new(pr_m.0, pr_m.1).map_err(to_pyerr)?,
            pr_k: Gamma::new(pr_k.0, pr_k.1).map_err(to_pyerr)?,
            pr_v: InvGamma::new(pr_v.0, pr_v.1).map_err(to_pyerr)?,
            pr_s2: InvGamma::new(pr_s2.0, pr_s2.1).map_err(to_pyerr)?,
        }))
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct CategoricalPrior(pub SymmetricDirichlet);

#[derive(Clone, Debug)]
#[pyclass]
pub struct CategoricalHyper(pub CsdHyper);

#[pymethods]
impl CategoricalPrior {
    #[new]
    #[pyo3(signature = (k, alpha=0.5))]
    pub fn new(k: usize, alpha: f64) -> PyResult<Self> {
        let inner = SymmetricDirichlet::new(alpha, k).map_err(to_pyerr)?;
        Ok(Self(inner))
    }
}

#[pymethods]
impl CategoricalHyper {
    #[new]
    #[pyo3(signature = (shape=1.0, scale=1.0))]
    pub fn new(shape: f64, scale: f64) -> PyResult<Self> {
        Ok(Self(CsdHyper {
            pr_alpha: InvGamma::new(shape, scale).map_err(to_pyerr)?,
        }))
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct ValueMap(lace::codebook::ValueMap);

#[pymethods]
impl ValueMap {
    #[classmethod]
    #[pyo3(signature = (n_cats))]
    pub fn int(_cls: &PyType, n_cats: usize) -> Self {
        Self(lace::codebook::ValueMap::U8(n_cats))
    }

    #[classmethod]
    #[pyo3(signature = (values))]
    pub fn string(_cls: &PyType, values: Vec<String>) -> PyResult<Self> {
        lace::codebook::ValueMap::try_from(values)
            .map_err(PyValueError::new_err)
            .map(Self)
    }

    #[classmethod]
    pub fn bool(_cls: &PyType) -> Self {
        Self(lace::codebook::ValueMap::Bool)
    }
}

#[pymethods]
impl ColumnMetadata {
    #[classmethod]
    #[pyo3(signature = (name, prior=None, hyper=None))]
    pub fn continuous(
        _cls: &PyType,
        name: String,
        prior: Option<ContinuousPrior>,
        hyper: Option<ContinuousHyper>,
    ) -> Self {
        Self(ColMetadata {
            name,
            coltype: ColType::Continuous {
                hyper: hyper.map(|hy| hy.0),
                prior: prior.map(|pr| pr.0),
            },
            notes: None,
            missing_not_at_random: false,
        })
    }

    #[classmethod]
    #[pyo3(signature = (name, k, value_map=None, prior=None, hyper=None))]
    pub fn categorical(
        _cls: &PyType,
        name: String,
        k: usize,
        value_map: Option<ValueMap>,
        prior: Option<CategoricalPrior>,
        hyper: Option<CategoricalHyper>,
    ) -> Self {
        Self(ColMetadata {
            name,
            coltype: ColType::Categorical {
                k,
                hyper: hyper.map(|hy| hy.0),
                prior: prior.map(|pr| pr.0),
                value_map: value_map
                    .map(|pr| pr.0)
                    .unwrap_or_else(|| lace::codebook::ValueMap::U8(k)),
            },
            notes: None,
            missing_not_at_random: false,
        })
    }

    pub fn missing_not_at_random(&self, mnar: bool) -> Self {
        let mut out = self.clone();
        out.0.missing_not_at_random = mnar;
        out
    }

    pub fn notes(&self, notes: Option<String>) -> Self {
        let mut out = self.clone();
        out.0.notes = notes;
        out
    }
}

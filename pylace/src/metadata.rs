use lace::codebook::{ColMetadata, ColType};
use lace::stats::prior::nix::NixHyper;
use lace::stats::rv::dist::NormalInvChiSquared;
use lace::stats::rv::prelude::{Gamma, Gaussian, InvGamma};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::utils::to_pyerr;

#[derive(Clone, Debug)]
#[pyclass]
pub struct ColumnMetadata(pub ColMetadata);

#[derive(Clone, Debug)]
#[pyclass]
pub struct ContinuousPrior(pub NormalInvChiSquared);

#[derive(Clone, Debug)]
#[pyclass]
pub struct ContinuousHyper(pub NixHyper);

#[pymethods]
impl ContinuousPrior {
    #[new]
    #[pyo3(signature = (m=0.0, k=1.0, v=1.0, s2=1.0))]
    pub fn new(m: f64, k: f64, v: f64, s2: f64) -> PyResult<Self> {
        let inner = NormalInvChiSquared::new(m, k, v, s2).map_err(to_pyerr)?;
        Ok(Self(inner))
    }
}

#[pymethods]
impl ContinuousHyper {
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
            name: name.into(),
            coltype: ColType::Continuous {
                hyper: hyper.map(|hy| hy.0),
                prior: prior.map(|pr| pr.0),
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

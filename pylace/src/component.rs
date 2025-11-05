use lace::rv::dist::{Bernoulli, Categorical, Gaussian, Poisson};
use lace::stats::MixtureType;
use pyo3::prelude::*;

impl From<MixtureType> for ComponentParams {
    fn from(mixture: MixtureType) -> Self {
        match mixture {
            MixtureType::Bernoulli(mm) => mm.components().as_slice().into(),
            MixtureType::Categorical(mm) => mm.components().as_slice().into(),
            MixtureType::Gaussian(mm) => mm.components().as_slice().into(),
            MixtureType::Poisson(mm) => mm.components().as_slice().into(),
        }
    }
}

pub enum ComponentParams {
    Bernoulli(Vec<BernoulliParams>),
    Categorical(Vec<CategoricalParams>),
    Gaussian(Vec<GaussianParams>),
    Poisson(Vec<PoissonParams>),
}

#[pyclass(get_all)]
pub struct BernoulliParams {
    pub p: f64,
}

#[pymethods]
impl BernoulliParams {
    pub fn __repr__(&self) -> String {
        format!("Bernoulli(p={})", self.p)
    }
}

#[pyclass(get_all)]
pub struct GaussianParams {
    pub mu: f64,
    pub sigma: f64,
}

#[pymethods]
impl GaussianParams {
    pub fn __repr__(&self) -> String {
        format!("Gaussian(mu={}, sigma={})", self.mu, self.sigma)
    }
}

#[pyclass(get_all)]
pub struct PoissonParams {
    pub rate: f64,
}

#[pymethods]
impl PoissonParams {
    pub fn __repr__(&self) -> String {
        format!("Poisson(rate={})", self.rate)
    }
}

#[pyclass(get_all)]
pub struct CategoricalParams {
    pub weights: Vec<f64>,
}

#[pymethods]
impl CategoricalParams {
    pub fn __repr__(&self) -> String {
        let k = self.weights.len();
        let weights_str = match k {
            0 => "[]".to_string(),
            1 => "[1.0]".to_string(),
            2 => format!("[{}, {}]", self.weights[0], self.weights[1]),
            _ => format!(
                "[{}, ..., {}]",
                self.weights[0],
                self.weights
                    .last()
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "-".to_string())
            ),
        };

        format!(
            "Categorical_{}(weights={})",
            self.weights.len(),
            weights_str
        )
    }
}

impl From<&Bernoulli> for BernoulliParams {
    fn from(dist: &Bernoulli) -> Self {
        BernoulliParams { p: dist.p() }
    }
}

impl From<&Categorical> for CategoricalParams {
    fn from(dist: &Categorical) -> Self {
        CategoricalParams {
            weights: dist.weights(),
        }
    }
}

impl From<&Gaussian> for GaussianParams {
    fn from(dist: &Gaussian) -> Self {
        GaussianParams {
            mu: dist.mu(),
            sigma: dist.sigma(),
        }
    }
}

impl From<&Poisson> for PoissonParams {
    fn from(dist: &Poisson) -> Self {
        PoissonParams { rate: dist.rate() }
    }
}

macro_rules! impl_from_slice_dist {
    ($variant: ident) => {
        impl From<&[$variant]> for ComponentParams {
            fn from(dists: &[$variant]) -> Self {
                let inner = dists.iter().map(|dist| dist.into()).collect();
                ComponentParams::$variant(inner)
            }
        }
    };
}

impl_from_slice_dist!(Bernoulli);
impl_from_slice_dist!(Gaussian);
impl_from_slice_dist!(Categorical);
impl_from_slice_dist!(Poisson);

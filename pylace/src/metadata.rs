use crate::df::PyDataFrame;
use lace::codebook::data::df_to_codebook;
use lace::codebook::{ColMetadata, ColMetadataList, ColType, RowNameList};
use lace::stats::prior::csd::CsdHyper;
use lace::stats::prior::nix::NixHyper;
use lace::stats::prior::pg::PgHyper;
use lace::stats::rv::dist::{
    Beta, Gamma, Gaussian, InvGamma, NormalInvChiSquared, SymmetricDirichlet,
};
use polars::prelude::DataFrame;
use pyo3::exceptions::{PyIOError, PyIndexError};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::path::PathBuf;

use crate::utils::to_pyerr;

macro_rules! newtype_json_repr {
    ($self: ident) => {{
        serde_json::to_string_pretty(&$self.0).map_err(to_pyerr)
    }};
}

macro_rules! newtype_string_repr {
    ($self: ident) => {{
        Ok($self.0.to_string())
    }};
}

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
    ///     How strongly we believe the prior mean (in prior pseudo-
    ///     observations)
    /// v: float
    ///     How strongly we believe the prior variance (in prior pseudo-
    ///     observations)
    /// s2: float
    ///     The prior variance
    #[new]
    #[pyo3(signature = (m=0.0, k=1.0, v=1.0, s2=1.0))]
    pub fn new(m: f64, k: f64, v: f64, s2: f64) -> PyResult<Self> {
        let inner = NormalInvChiSquared::new(m, k, v, s2).map_err(to_pyerr)?;
        Ok(Self(inner))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_string_repr!(self)
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

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_json_repr!(self)
    }
}

/// Prior on categorical data
#[pyclass]
#[derive(Clone, Debug)]
pub struct CategoricalPrior(pub SymmetricDirichlet);

/// Hyperprior on categorical prior
#[pyclass]
#[derive(Clone, Debug)]
pub struct CategoricalHyper(pub CsdHyper);

#[pymethods]
impl CategoricalPrior {
    /// Create a new ``CategoricalPrior``
    ///
    /// Parameters
    /// ----------
    /// k: int
    ///     The number of categories
    /// alpha: float
    ///     The symmetric Dirichlet weight
    #[new]
    #[pyo3(signature = (k, alpha=0.5))]
    pub fn new(k: usize, alpha: f64) -> PyResult<Self> {
        let inner = SymmetricDirichlet::new(alpha, k).map_err(to_pyerr)?;
        Ok(Self(inner))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_string_repr!(self)
    }
}

#[pymethods]
impl CategoricalHyper {
    /// Create a new ``CategoricalHyper``
    ///
    /// Parameters
    /// ----------
    /// shape: float
    ///     The inverse gamma shape parameter
    /// scale: float
    ///     The inverse gamma scale parameter
    #[new]
    #[pyo3(signature = (shape=1.0, scale=1.0))]
    pub fn new(shape: f64, scale: f64) -> PyResult<Self> {
        Ok(Self(CsdHyper {
            pr_alpha: InvGamma::new(shape, scale).map_err(to_pyerr)?,
        }))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_json_repr!(self)
    }
}

/// Prior on count data
#[pyclass]
#[derive(Clone, Debug)]
pub struct CountPrior(pub Gamma);

/// Hyperprior on categorical prior
#[pyclass]
#[derive(Clone, Debug)]
pub struct CountHyper(pub PgHyper);

#[pymethods]
impl CountPrior {
    /// Create a new ``CountPrior``
    ///
    /// Parameters
    /// ----------
    /// shape: float
    ///     The gamma shape parameter
    /// rate: float
    ///     The gamma rate parameter
    #[new]
    #[pyo3(signature = (shape=1.0, rate=1.0))]
    pub fn new(shape: f64, rate: f64) -> PyResult<Self> {
        let inner = Gamma::new(shape, rate).map_err(to_pyerr)?;
        Ok(Self(inner))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_string_repr!(self)
    }
}

#[pymethods]
impl CountHyper {
    /// Create a new ``CountHyper``
    ///
    /// Parameters
    /// ----------
    /// pr_shape: (float, float)
    ///     The shape and rate gamma parameters on the prior shape
    /// pr_rate: (float, float)
    ///     The shape and scale inverse gamma parameters on the prior rate
    #[new]
    #[pyo3(signature = (pr_shape=(1.0, 1.0), pr_rate=(1.0, 1.0)))]
    pub fn new(pr_shape: (f64, f64), pr_rate: (f64, f64)) -> PyResult<Self> {
        Ok(Self(PgHyper {
            pr_shape: Gamma::new(pr_shape.0, pr_shape.1).map_err(to_pyerr)?,
            pr_rate: InvGamma::new(pr_rate.0, pr_rate.1).map_err(to_pyerr)?,
        }))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_json_repr!(self)
    }
}

/// A map of categorical values to unsigned integers
#[pyclass]
#[derive(Clone, Debug)]
pub struct ValueMap(lace::codebook::ValueMap);

#[pyclass]
pub struct ValueMapIterator {
    inner: lace::codebook::ValueMap,
    ix: usize,
}

#[pymethods]
impl ValueMapIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyObject> {
        use lace::codebook::ValueMap as Vm;
        match slf.inner {
            Vm::String(ref map) => {
                if slf.ix >= map.len() {
                    None
                } else {
                    let s = map.category(slf.ix);
                    slf.ix += 1;
                    Some(s.into_py(slf.py()))
                }
            }
            Vm::UInt(k) => {
                if slf.ix >= k {
                    None
                } else {
                    let c = slf.ix.into_py(slf.py());
                    slf.ix += 1;
                    Some(c)
                }
            }
            Vm::Bool => {
                if slf.ix == 0 {
                    slf.ix += 1;
                    Some(false.into_py(slf.py()))
                } else if slf.ix == 1 {
                    slf.ix += 1;
                    Some(true.into_py(slf.py()))
                } else {
                    None
                }
            }
        }
    }
}

#[pymethods]
impl ValueMap {
    /// Create a map of ``k`` unsigned integers
    #[classmethod]
    #[pyo3(signature = (k))]
    pub fn int(_cls: &Bound<PyType>, k: usize) -> Self {
        Self(lace::codebook::ValueMap::UInt(k))
    }

    /// Create a map from strings
    ///
    /// Parameters
    /// ----------
    /// values: list[str]
    ///     A list of unique strings
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     The strings are not unique
    #[classmethod]
    #[pyo3(signature = (values))]
    pub fn string(_cls: &Bound<PyType>, values: Vec<String>) -> PyResult<Self> {
        lace::codebook::ValueMap::try_from(values)
            .map_err(PyValueError::new_err)
            .map(Self)
    }

    /// Create a map from boolean
    #[classmethod]
    pub fn bool(_cls: &Bound<PyType>) -> Self {
        Self(lace::codebook::ValueMap::Bool)
    }

    pub fn __repr__(&self) -> String {
        use lace::codebook::ValueMap as Vm;
        match self.0 {
            Vm::UInt(k) => format!("ValueMap (UInt, k={k})"),
            Vm::String(ref inner) => {
                let k = inner.len();
                let cats = (0..k)
                    .map(|ix| format!("'{}' ", inner.category(ix)))
                    .collect::<String>();
                format!("ValueMap (String) [ {cats}]")
            }
            Vm::Bool => String::from("ValueMap (bool)"),
        }
    }

    fn values(slf: PyRef<'_, Self>) -> PyResult<Py<ValueMapIterator>> {
        let iter = ValueMapIterator {
            inner: slf.0.clone(),
            ix: 0,
        };

        Py::new(slf.py(), iter)
    }
}

#[pymethods]
impl ColumnMetadata {
    /// Create continuous column metadata
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the column
    /// prior: ContinuousPrior, optional
    ///     The prior on the data. If ``None`` (default), a prior will be drawn
    ///     from the ``hyper``
    /// hyper: ContinuousHyper, optional
    ///     The prior on the data. If ``None`` (default) and ``prior`` is
    ///     defined, the prior parameters will be locked.
    #[classmethod]
    #[pyo3(signature = (name, prior=None, hyper=None))]
    pub fn continuous(
        _cls: &Bound<PyType>,
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

    /// Create categorical column metadata
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the column
    /// k: int
    ///     The number of categories
    /// value_map: ValueMap, optional
    ///     A map from possible values to unsigned integers. If ``None``
    ///     (default), it is assumed the values to this column take on values
    ///     in [0, k-1].
    /// prior: CategoricalPrior, optional
    ///     The prior on the data. If ``None`` (default), a prior will be drawn
    ///     from the ``hyper``
    /// hyper: CategoricalHyper, optional
    ///     The prior on the data. If ``None`` (default) and ``prior`` is
    ///     defined, the prior parameters will be locked.
    #[classmethod]
    #[pyo3(signature = (name, k, value_map=None, prior=None, hyper=None))]
    pub fn categorical(
        _cls: &Bound<PyType>,
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
                    .unwrap_or_else(|| lace::codebook::ValueMap::UInt(k)),
            },
            notes: None,
            missing_not_at_random: false,
        })
    }

    /// Create count column metadata
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the column
    /// prior: CountPrior, optional
    ///     The prior on the data. If ``None`` (default), a prior will be drawn
    ///     from the ``hyper``
    /// hyper: CountHyper, optional
    ///     The prior on the data. If ``None`` (default) and ``prior`` is
    ///     defined, the prior parameters will be locked.
    #[classmethod]
    #[pyo3(signature = (name, prior=None, hyper=None))]
    pub fn count(
        _cls: &Bound<PyType>,
        name: String,
        prior: Option<CountPrior>,
        hyper: Option<CountHyper>,
    ) -> Self {
        Self(ColMetadata {
            name,
            coltype: ColType::Count {
                hyper: hyper.map(|hy| hy.0),
                prior: prior.map(|pr| pr.0),
            },
            notes: None,
            missing_not_at_random: false,
        })
    }

    /// Set whether the column missing data should be modeled as non-random
    pub fn missing_not_at_random(&self, mnar: bool) -> Self {
        let mut out = self.clone();
        out.0.missing_not_at_random = mnar;
        out
    }

    /// Add notes
    pub fn notes(&self, notes: Option<String>) -> Self {
        let mut out = self.clone();
        out.0.notes = notes;
        out
    }

    /// Rename
    pub fn rename(&self, name: String) -> Self {
        let mut out = self.clone();
        out.0.name = name;
        out
    }

    #[getter]
    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    pub fn __repr__(&self) -> PyResult<String> {
        newtype_json_repr!(self)
    }

    #[getter]
    pub fn value_map(&self) -> Option<ValueMap> {
        match self.0.coltype {
            ColType::Categorical { ref value_map, .. } => {
                Some(ValueMap(value_map.clone()))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
enum CodebookMethod {
    Path(PathBuf),
    Inferred {
        cat_cutoff: Option<u32>,
        state_prior_process: Option<PriorProcess>,
        view_prior_process: Option<PriorProcess>,
        no_hypers: bool,
    },
    Codebook(Codebook),
}

impl Default for CodebookMethod {
    fn default() -> Self {
        Self::Inferred {
            cat_cutoff: None,
            state_prior_process: None,
            view_prior_process: None,
            no_hypers: false,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PriorProcess(lace::codebook::PriorProcess);

#[pymethods]
impl PriorProcess {
    #[classmethod]
    #[pyo3(signature=(alpha_shape=1.0, alpha_rate=1.0, d_a=0.5, d_b=0.5))]
    pub fn pitman_yor(
        _cls: &Bound<PyType>,
        alpha_shape: f64,
        alpha_rate: f64,
        d_a: f64,
        d_b: f64,
    ) -> Self {
        PriorProcess(lace::codebook::PriorProcess::PitmanYor {
            alpha_prior: Gamma::new(alpha_shape, alpha_rate).unwrap(),
            d_prior: Beta::new(d_a, d_b).unwrap(),
        })
    }

    #[classmethod]
    #[pyo3(signature=(alpha_shape=1.0, alpha_rate=1.0))]
    pub fn dirichlet(
        _cls: &Bound<PyType>,
        alpha_shape: f64,
        alpha_rate: f64,
    ) -> Self {
        PriorProcess(lace::codebook::PriorProcess::Dirichlet {
            alpha_prior: Gamma::new(alpha_shape, alpha_rate).unwrap(),
        })
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Codebook(pub(crate) lace::codebook::Codebook);

#[pymethods]
impl Codebook {
    #[new]
    pub fn new(table_name: String) -> Self {
        let mut codebook = lace::codebook::Codebook::default();
        codebook.table_name = table_name;
        Self(codebook)
    }

    pub fn rename(&mut self, table_name: String) {
        self.0.table_name = table_name;
    }

    pub fn set_state_prior_process(
        &mut self,
        process: PriorProcess,
    ) -> PyResult<()> {
        self.0.state_prior_process = Some(process.0);
        Ok(())
    }

    pub fn set_view_prior_process(
        &mut self,
        process: PriorProcess,
    ) -> PyResult<()> {
        self.0.view_prior_process = Some(process.0);
        Ok(())
    }

    pub fn set_row_names(&mut self, row_names: Vec<String>) -> PyResult<()> {
        let row_names: RowNameList = row_names.try_into().map_err(to_pyerr)?;
        self.0.row_names = row_names;
        Ok(())
    }

    pub fn append_column_metadata(
        &mut self,
        mut col_metadata: Vec<ColumnMetadata>,
    ) -> PyResult<()> {
        let col_metadata: ColMetadataList = col_metadata
            .drain(..)
            .map(|md| md.0)
            .collect::<Vec<ColMetadata>>()
            .try_into()
            .map_err(to_pyerr)?;
        self.0.append_col_metadata(col_metadata).map_err(to_pyerr)
    }

    pub fn remove_column_metadata(
        &mut self,
        name: String,
    ) -> PyResult<ColumnMetadata> {
        self.0
            .col_metadata
            .take(name.as_str())
            .ok_or_else(|| PyKeyError::new_err(format!("No '{name}' column")))
            .map(ColumnMetadata)
    }

    #[pyo3(signature = (pretty=true))]
    pub fn json(&self, pretty: bool) -> PyResult<String> {
        if pretty {
            serde_json::to_string_pretty(&self.0)
        } else {
            serde_json::to_string(&self.0)
        }
        .map_err(to_pyerr)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Codebook '{}'\
            \n  state_prior_process: {}\
            \n  view_prior_process: {}\
            \n  columns: {}\
            \n  rows: {}",
            self.0.table_name,
            self.0
                .state_prior_process
                .clone()
                .map_or_else(|| String::from("None"), |p| format!("{}", p)),
            self.0
                .view_prior_process
                .clone()
                .map_or_else(|| String::from("None"), |p| format!("{}", p)),
            self.0.col_metadata.len(),
            self.0.row_names.len()
        )
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.0.row_names.len(), self.0.col_metadata.len())
    }

    #[getter]
    fn row_names(&self) -> Vec<String> {
        self.0.row_names.clone().into()
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.0
            .col_metadata
            .iter()
            .map(|md| md.name.clone())
            .collect()
    }

    fn column_metadata(&self, name: &str) -> PyResult<ColumnMetadata> {
        self.0
            .col_metadata
            .get(name)
            .ok_or_else(|| PyIndexError::new_err(format!("No column '{name}'")))
            .map(|(_, md)| ColumnMetadata(md.clone()))
    }

    fn set_column_metadata(
        &mut self,
        name: &str,
        col_metadata: ColumnMetadata,
    ) -> PyResult<()> {
        if self.0.column_index(name).is_none() {
            Err(PyKeyError::new_err(format!(
                "No '{name}' column in codebook"
            )))
        } else {
            self.0.col_metadata[name] = col_metadata.0;
            Ok(())
        }
    }

    /// Create a new codebook with the same columns but row indices from another dataframe.
    fn with_index(&self, new_index: Vec<String>) -> PyResult<Self> {
        let row_names = RowNameList::try_from(new_index).map_err(to_pyerr)?;
        Ok(Self(lace::codebook::Codebook {
            row_names,
            ..self.0.clone()
        }))
    }
}

#[pyfunction]
#[pyo3(signature = (df, cat_cutoff=None, no_hypers=false))]
pub fn codebook_from_df(
    df: PyDataFrame,
    cat_cutoff: Option<u32>,
    no_hypers: bool,
) -> PyResult<Codebook> {
    CodebookBuilder {
        method: CodebookMethod::Inferred {
            cat_cutoff,
            state_prior_process: None,
            view_prior_process: None,
            no_hypers,
        },
    }
    .build(&df.0)
    .map(Codebook)
}

#[derive(Clone, Debug, Default)]
#[pyclass]
pub struct CodebookBuilder {
    method: CodebookMethod,
}

#[pymethods]
impl CodebookBuilder {
    #[classmethod]
    /// Load a Codebook from a path.
    fn load(_cls: &Bound<PyType>, path: PathBuf) -> Self {
        Self {
            method: CodebookMethod::Path(path),
        }
    }

    #[classmethod]
    #[pyo3(signature = (cat_cutoff=None, state_prior_process=None, view_prior_process=None, use_hypers=true))]
    fn infer(
        _cls: &Bound<PyType>,
        cat_cutoff: Option<u32>,
        state_prior_process: Option<PriorProcess>,
        view_prior_process: Option<PriorProcess>,
        use_hypers: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            method: CodebookMethod::Inferred {
                cat_cutoff,
                state_prior_process,
                view_prior_process,
                no_hypers: !use_hypers,
            },
        })
    }

    #[classmethod]
    fn codebook(_cls: &Bound<PyType>, codebook: Codebook) -> Self {
        Self {
            method: CodebookMethod::Codebook(codebook),
        }
    }
}

impl CodebookBuilder {
    /// Create a codebook from the method described.
    pub fn build(self, df: &DataFrame) -> PyResult<lace::codebook::Codebook> {
        match self.method {
            CodebookMethod::Path(path) => {
                let file =
                    std::fs::File::open(path.clone()).map_err(|err| {
                        PyIOError::new_err(format!(
                            "Error opening {path:?}: {err}",
                        ))
                    })?;

                serde_yaml::from_reader(&file)
                    .or_else(|_| serde_json::from_reader(&file))
                    .map_err(|_| {
                        PyIOError::new_err(format!(
                            "Failed to read codebook at {path:?}"
                        ))
                    })
            }
            CodebookMethod::Inferred {
                cat_cutoff,
                state_prior_process,
                view_prior_process,
                no_hypers,
            } => df_to_codebook(
                df,
                cat_cutoff,
                state_prior_process.map(|p| p.0),
                view_prior_process.map(|p| p.0),
                no_hypers,
            )
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to infer the Codebook. Error: {e}"
                ))
            }),
            CodebookMethod::Codebook(codebook) => Ok(codebook.0),
        }
    }

    fn __repr__(&self) -> String {
        match &self.method {
            CodebookMethod::Path(path) => format!("<CodebookBuilder path='{}'>", path.display()),
            CodebookMethod::Inferred { cat_cutoff, state_prior_process, view_prior_process, no_hypers } => format!("CodebookBuilder Inferred(cat_cutoff={cat_cutoff:?}, state_prior_process={state_prior_process:?}, view_prior_process={view_prior_process:?}, use_hypers={})", !no_hypers),
            CodebookMethod::Codebook(_) => String::from("Codebook (fully specified)"),
        }
    }
}

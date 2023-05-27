mod df;
mod metadata;
mod transition;
mod utils;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;

use df::{DataFrameLike, PyDataFrame, PySeries};
use lace::codebook::data::df_to_codebook;
use lace::codebook::Codebook;
use lace::data::DataSource;
use lace::metadata::SerializedType;
use lace::prelude::ColMetadataList;
use lace::stats::rv::prelude::Gamma;
use lace::{
    EngineUpdateConfig, FType, HasStates, OracleT, PredictUncertaintyType,
};
use metadata::ColumnMetadata;
use polars::prelude::{DataFrame, NamedFrom, Series};
use pyo3::exceptions::{PyIOError, PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PySlice, PyString, PyType};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::utils::*;

#[derive(Clone, Debug)]
enum CodebookMethod {
    Path(PathBuf),
    Inferred {
        cat_cutoff: Option<u8>,
        alpha_prior_opt: Option<Gamma>,
        no_hypers: bool,
    },
}

impl Default for CodebookMethod {
    fn default() -> Self {
        Self::Inferred {
            cat_cutoff: None,
            alpha_prior_opt: None,
            no_hypers: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
#[pyclass]
struct CodebookBuilder {
    method: CodebookMethod,
}

#[pymethods]
impl CodebookBuilder {
    #[classmethod]
    /// Load a Codebook from a path.
    fn load(_cls: &PyType, path: PathBuf) -> Self {
        Self {
            method: CodebookMethod::Path(path),
        }
    }

    #[classmethod]
    #[pyo3(signature = (cat_cutoff=None, alpha_prior_shape_rate=None, use_hypers=true))]
    fn infer(
        _cls: &PyType,
        cat_cutoff: Option<u8>,
        alpha_prior_shape_rate: Option<(f64, f64)>,
        use_hypers: bool,
    ) -> PyResult<Self> {
        let alpha_prior_opt = alpha_prior_shape_rate
            .map(|(shape, rate)| Gamma::new(shape, rate))
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            method: CodebookMethod::Inferred {
                cat_cutoff,
                alpha_prior_opt,
                no_hypers: !use_hypers,
            },
        })
    }
}

impl CodebookBuilder {
    /// Create a codebook from the method described.
    pub fn build(self, df: &DataFrame) -> PyResult<Codebook> {
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
                alpha_prior_opt,
                no_hypers,
            } => df_to_codebook(df, cat_cutoff, alpha_prior_opt, no_hypers)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to infer the Codebook. Error: {e}"
                    ))
                }),
        }
    }

    fn __repr__(&self) -> String {
        match &self.method {
            CodebookMethod::Path(path) => format!("<CodebookBuilder path='{}'>", path.display()),
            CodebookMethod::Inferred { cat_cutoff, alpha_prior_opt, no_hypers } => format!("CodebookBuilder Inferred(cat_cutoff={cat_cutoff:?}, alpha_prior_opt={alpha_prior_opt:?}, use_hypers={})", !no_hypers),
        }
    }
}

#[pyclass(subclass)]
struct CoreEngine {
    engine: lace::Engine,
    col_indexer: Indexer,
    row_indexer: Indexer,
    rng: Xoshiro256Plus,
}

// FIXME: implement __repr__
// FIXME: implement name (get name from codebook)
#[pymethods]
impl CoreEngine {
    /// Create a new Engine from the prior
    ///
    /// Parameters
    /// ----------
    /// dataframe: DataFrame
    ///     polars DataFrame with subject data.
    /// codebook_builder: CodebookBuilder, optional
    ///     Optional codebook builder from
    /// n_states: int, optional
    ///     The number of states (posterior samples)
    /// id_offset: int, optional
    ///     Increase the IDs of the states by an integer. Used when you want to
    ///     merge the states in multiple metadata files. For example, if you run
    ///     4 states on two separate machine, you would use an `id_offset` of 2
    ///     on the second machine so that the state files have names `2.state`
    ///     and `3.state`
    /// rng_seed: int, optional
    ///     Integer seed for the random number generator
    /// source_type: str, optional
    ///     The type of the data file. Can be `'csv'`, `'csv.gz'`, `'feather'`,
    ///     `'ipc'`, `'parquet'`, `'json'`, or `'jsonl'`. If None (default) the
    ///     source type will be inferred from the `data_source` file extension.
    /// cat_cutoff: int, optional
    ///     The number of distinct unsigned integer values and column can assume
    ///     before it is inferred not to be a categorical type.
    /// no_hypers: bool, optional
    ///     If True, features hyperparameter inference will not be conducted.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(
        signature = (
            dataframe,
            codebook_builder=None,
            n_states=16,
            id_offset=0,
            rng_seed=None,
        )
    )]
    fn new(
        dataframe: PyDataFrame,
        codebook_builder: Option<CodebookBuilder>,
        n_states: usize,
        id_offset: usize,
        rng_seed: Option<u64>,
    ) -> PyResult<CoreEngine> {
        let dataframe = dataframe.0;
        let codebook =
            codebook_builder.unwrap_or_default().build(&dataframe)?;
        let data_source = DataSource::Polars(dataframe);

        let rng = if let Some(seed) = rng_seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let engine = lace::Engine::new(
            n_states,
            codebook,
            data_source,
            id_offset,
            rng.clone(),
        )
        .map_err(|err| err.to_string())
        .map_err(PyErr::new::<PyValueError, _>)?;

        Ok(Self {
            col_indexer: Indexer::columns(&engine.codebook),
            row_indexer: Indexer::rows(&engine.codebook),
            rng,
            engine,
        })
    }

    /// Load a Engine from metadata
    #[classmethod]
    fn load(_cls: &PyType, path: PathBuf) -> CoreEngine {
        let (engine, rng) = {
            let mut engine = lace::Engine::load(path).unwrap();
            let rng = Xoshiro256Plus::from_rng(&mut engine.rng).unwrap();
            (engine, rng)
        };
        Self {
            col_indexer: Indexer::columns(&engine.codebook),
            row_indexer: Indexer::rows(&engine.codebook),
            rng,
            engine,
        }
    }

    /// Save the engine to `path`
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.engine
            .save(path, SerializedType::Bincode)
            .map_err(to_pyerr)
    }

    /// Seed the random number generator
    fn seed(&mut self, rng_seed: u64) {
        self.rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    }

    /// Return the number of states
    #[getter]
    fn n_states(&self) -> usize {
        self.engine.n_states()
    }

    /// Return the number of rows
    #[getter]
    fn n_rows(&self) -> usize {
        self.engine.n_rows()
    }

    /// Return the number of columns
    #[getter]
    fn n_cols(&self) -> usize {
        self.engine.n_cols()
    }

    /// Return the (n_rows, n_cols) shape of the table
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.engine.n_rows(), self.engine.n_cols())
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.engine
            .codebook
            .col_metadata
            .iter()
            .map(|col_md| col_md.name.clone())
            .collect()
    }

    #[getter]
    fn index(&self) -> Vec<String> {
        self.engine.codebook.row_names.as_slice().to_owned()
    }

    fn __getitem__(&self, ixs: utils::TableIndex) -> PyResult<PyDataFrame> {
        let (row_ixs, col_ixs) = ixs.ixs(&self.engine.codebook)?;

        let index = polars::series::Series::new(
            "Index",
            row_ixs
                .iter()
                .map(|ix| ix.1.clone())
                .collect::<Vec<String>>(),
        );

        let mut df = polars::frame::DataFrame::empty();
        df.with_column(index).map_err(to_pyerr)?;

        for col_ix in &col_ixs {
            let mut values = Vec::new();
            for row_ix in &row_ixs {
                let value =
                    self.engine.datum(row_ix.0, col_ix.0).map_err(to_pyerr)?;
                values.push(value);
            }
            let ftype = self.engine.ftype(col_ix.0).map_err(to_pyerr)?;
            let srs = utils::vec_to_srs(
                values,
                col_ix.0,
                ftype,
                &self.engine.codebook,
            )?;
            df.with_column(srs.0).map_err(to_pyerr)?;
        }
        Ok(PyDataFrame(df))
    }

    #[getter]
    fn ftypes(&self) -> HashMap<String, String> {
        self.engine
            .ftypes()
            .drain(..)
            .enumerate()
            .map(|(col_ix, ftype)| {
                let col_name = self.col_indexer.to_name[&col_ix].to_owned();
                let ftype_str = ftype.to_string();
                (col_name, ftype_str)
            })
            .collect()
    }

    fn ftype(&self, col: &PyAny) -> PyResult<String> {
        let col_ix = utils::value_to_name(col, &self.col_indexer)?;
        self.engine
            .ftype(col_ix)
            .map_err(|err| {
                PyErr::new::<PyIndexError, _>(format!(
                    "Failed to get ftype: {err}"
                ))
            })
            .map(String::from)
    }

    fn column_assignment(&self, state_ix: usize) -> PyResult<Vec<usize>> {
        let n_states = self.n_states();
        if state_ix >= n_states {
            let msg = format!(
                "state index {state_ix} is out of bounds for  engine with \
                {n_states} states"
            );
            Err(PyErr::new::<PyIndexError, _>(msg))
        } else {
            Ok(self.engine.states[state_ix].asgn.asgn.clone())
        }
    }

    fn row_assignments(&self, state_ix: usize) -> PyResult<Vec<Vec<usize>>> {
        let n_states = self.n_states();
        if state_ix >= n_states {
            let msg = format!(
                "state index {state_ix} is out of bounds for  engine with \
                {n_states} states"
            );
            Err(PyErr::new::<PyIndexError, _>(msg))
        } else {
            let asgns = self.engine.states[state_ix]
                .views
                .iter()
                .map(|view| view.asgn.asgn.clone())
                .collect();
            Ok(asgns)
        }
    }

    fn diagnostics(&self) -> Vec<HashMap<String, Vec<f64>>> {
        (0..self.n_states())
            .map(|ix| {
                let mut diag = HashMap::new();
                diag.insert(
                    String::from("loglike"),
                    self.engine.states[ix].diagnostics.loglike.clone(),
                );
                diag.insert(
                    String::from("logprior"),
                    self.engine.states[ix].diagnostics.logprior.clone(),
                );
                diag
            })
            .collect()
    }

    /// Dependence probability
    ///
    /// Parameters
    /// ----------
    /// col_pairs: list((index-like, index-like))
    ///     A list of column pairs over which to compute dependence probability. Integers are
    ///     treated like indices and string are treated as names.
    ///
    /// Returns
    /// -------
    /// depprob: numpy.array(float)
    ///     The dependence probability for every entry in `col_pairs`
    ///
    /// Example
    /// -------
    ///
    /// >>> import lace
    /// >>> engine = lace.Engine('animals.rp')
    /// >>> engine.depprob([(0, 1), (1, 2)])
    /// array([0.125, 0.   ])
    /// >>> engine.depprob([('swims', 'flippers'), ('swims', 'fast')])
    /// array([0.875, 0.25 ])
    fn depprob(&self, col_pairs: &PyList) -> PyResult<PySeries> {
        let pairs = list_to_pairs(col_pairs, &self.col_indexer)?;
        self.engine
            .depprob_pw(&pairs)
            .map_err(to_pyerr)
            .map(|xs| PySeries(Series::new("depprob", xs)))
    }

    /// Mutual information
    #[pyo3(signature=(col_pairs, n_mc_samples=1000, mi_type="iqr"))]
    fn mi(
        &self,
        col_pairs: &PyList,
        n_mc_samples: usize,
        mi_type: &str,
    ) -> PyResult<PySeries> {
        let pairs = list_to_pairs(col_pairs, &self.col_indexer)?;
        let mi_type = utils::str_to_mitype(mi_type)?;
        self.engine
            .mi_pw(&pairs, n_mc_samples, mi_type)
            .map_err(to_pyerr)
            .map(|xs| PySeries(Series::new("mi", xs)))
    }

    /// Row similarlity
    ///
    /// Parameters
    /// ----------
    /// row_pairs: list((index-like, index-like))
    ///     A list of row pairs over which to compute row similarity. Integers
    ///     are treated like indices and string are treated as names.
    /// wrt: list(index-like), optional
    ///     With respect to. A list of columns to contextualize row similarity.
    ///     The row similarity computation will be limited to the columns listed
    ///     in `wrt`. If `None` (default), all columns will be used.
    /// col_weighted: bool, optional
    ///     If `True`, row similarity will be weighted according to the number
    ///     of columns in each view. If `False` (default), row similarity will
    ///     be weighted according to the total number of views.
    ///
    /// Returns
    /// -------
    /// rowsim: numpy.array(float)
    ///     The row similarity for every pair in `row_pairs`
    ///
    /// Example
    /// ------
    ///
    /// >>> import lace
    /// >>> engine = Engine('animals.rp')
    /// >>> engine.rowsim([(0, 1), (1, 2)])
    /// array([0.51428571, 0.20982143])
    ///
    /// Using row names for indices
    ///
    /// >>> engine.rowsim([
    /// ...     ('grizzly+bear', 'bat'),
    /// ...     ('grizzly+bear', 'polar+bear')
    /// ... ])
    /// array([0.42767857, 0.52589286])
    ///
    /// Using the column-weighted variant
    ///
    /// >>> engine.rowsim(
    /// ...     [('grizzly+bear', 'bat'), ('grizzly+bear', 'polar+bear')],
    /// ...     col_weighted=True
    /// ... )
    /// array([0.22205882, 0.56764706])
    ///
    /// Adding context using `wrt` (with respect to). Compute the similarity
    /// with respect to how we predict whether these animals swim.
    ///
    /// >>> engine.rowsim(
    /// ...     [('grizzly+bear', 'bat'), ('grizzly+bear', 'polar+bear')],
    /// ...     wrt=['swims']
    /// ... )
    #[pyo3(signature=(row_pairs, wrt=None, col_weighted=false))]
    fn rowsim(
        &self,
        row_pairs: &PyList,
        wrt: Option<&PyAny>,
        col_weighted: bool,
    ) -> PyResult<PySeries> {
        let variant = if col_weighted {
            lace::RowSimilarityVariant::ColumnWeighted
        } else {
            lace::RowSimilarityVariant::ViewWeighted
        };

        let pairs = list_to_pairs(row_pairs, &self.row_indexer)?;
        if let Some(cols) = wrt {
            let wrt = pyany_to_indices(cols, &self.col_indexer)?;
            self.engine.rowsim_pw(&pairs, Some(wrt.as_slice()), variant)
        } else {
            self.engine.rowsim_pw::<_, usize>(&pairs, None, variant)
        }
        .map_err(to_pyerr)
        .map(|xs| PySeries(Series::new("rowsim", xs)))
    }

    #[pyo3(signature=(fn_name, pairs, fn_kwargs=None))]
    fn pairwise_fn(
        &self,
        fn_name: &str,
        pairs: &PyList,
        fn_kwargs: Option<&PyDict>,
    ) -> PyResult<PyDataFrame> {
        match fn_name {
            "depprob" => self.depprob(pairs).map(|xs| (xs, &self.col_indexer)),
            "mi" => {
                let args = fn_kwargs.map_or_else(
                    || Ok(utils::MiArgs::default()),
                    utils::mi_args_from_dict,
                )?;
                self.mi(pairs, args.n_mc_samples, args.mi_type.as_str())
                    .map(|xs| (xs, &self.col_indexer))
            }
            "rowsim" => {
                let args = fn_kwargs.map_or_else(
                    || Ok(utils::RowsimArgs::default()),
                    utils::rowsim_args_from_dict,
                )?;
                self.rowsim(pairs, args.wrt, args.col_weighted)
                    .map(|xs| (xs, &self.row_indexer))
            }
            _ => Err(PyErr::new::<PyValueError, _>(format!(
                "Unsupported pairwise fn: {}",
                fn_name
            ))),
        }
        .and_then(|(values, indexer)| {
            let mut a = Vec::with_capacity(pairs.len());
            let mut b = Vec::with_capacity(pairs.len());

            utils::pairs_list_iter(pairs, indexer).for_each(|res| {
                let (ix_a, ix_b) = res.unwrap();
                let name_a = indexer.to_name[&ix_a].clone();
                let name_b = indexer.to_name[&ix_b].clone();
                a.push(name_a);
                b.push(name_b);
            });

            let a = Series::new("A", a);
            let b = Series::new("B", b);

            polars::prelude::df!(
                "A" => a,
                "B" => b,
                fn_name => values.0,
            )
            .map_err(|err| {
                PyErr::new::<PyRuntimeError, _>(format!(
                    "Failed to create df: {}",
                    err
                ))
            })
            .map(PyDataFrame)
        })
    }

    /// Simulate data from a conditional distribution
    ///
    /// Parameters
    /// ----------
    /// cols: list(int or str)
    ///     The columns to simulate
    /// given: dict, optional
    ///     Column -> Value dictionary describing observations. Note that
    ///     columns can either be indices (int) or names (str)
    /// n: int, optional
    ///     The number of records to simulate (default: 1)
    ///
    /// Returns
    /// -------
    /// values: list(list(value))
    ///     `n` rows corresponding to a draw from each column in `cols`
    #[pyo3(signature = (cols, given=None, n=1))]
    fn simulate(
        &mut self,
        cols: &PyAny,
        given: Option<&PyDict>,
        n: usize,
    ) -> PyResult<PyDataFrame> {
        let col_ixs = pyany_to_indices(cols, &self.col_indexer)?;
        let given = dict_to_given(given, &self.engine, &self.col_indexer)?;

        let values = self
            .engine
            .simulate(&col_ixs, &given, n, None, &mut self.rng)
            .map_err(to_pyerr)?;

        utils::simulate_to_df(
            values,
            &self.engine.ftypes(),
            &col_ixs,
            &self.col_indexer,
            &self.engine.codebook,
        )
    }

    /// Draw data from the distribution of a specific cell in the table
    ///
    /// Parameters
    /// ----------
    /// row: str, int
    ///     The row name or index of the cell
    /// col: str, int
    ///     The column name or index of the cell
    /// n: int, optional
    ///     The number of samples to draw
    ///
    /// Returns
    /// -------
    /// srs: polars.Series
    ///     A polars Series with `n` entries
    #[pyo3(signature = (row, col, n=1))]
    fn draw(
        &mut self,
        row: &PyAny,
        col: &PyAny,
        n: usize,
    ) -> PyResult<PySeries> {
        let row_ix = utils::value_to_index(row, &self.row_indexer)?;
        let col_ix = utils::value_to_index(col, &self.col_indexer)?;
        let values = self
            .engine
            .draw(row_ix, col_ix, n, &mut self.rng)
            .map_err(to_pyerr)?;

        let ftype = self.engine.ftype(col_ix).map_err(to_pyerr)?;

        utils::vec_to_srs(values, col_ix, ftype, &self.engine.codebook)
    }

    /// Compute the log likelihood of data given optional conditions
    ///
    /// Parameters
    /// ----------
    /// values: pandas.DataFrame or pandas.Series
    ///     The input data. Each row is a record. Column names must match column
    ///     names in the lace table.
    /// given: dict, optional
    ///     Column -> Value dictionary describing observations. Note that
    ///     columns can either be indices (int) or names (str)
    /// state_ixs: list[int]
    ///     Optional list of states to use for computation. If `None` (default),
    ///     uses all states.
    ///
    /// Returns
    /// -------
    /// logp: numpy.array(float)
    ///     The log likelihood of each from in `values`
    ///
    /// Example
    /// -------
    ///
    /// >>> import pandas as pd
    /// >>> import lace
    /// >>>
    /// >>> engine = lace.Engine('animals.rp')
    /// >>>
    /// >>> xs = pd.DataFrame({
    /// ...     'swims': [0, 0, 1, 1],
    /// ...     'black': [0, 1, 0, 1]
    /// ... })
    /// >>>
    /// >>> engine.logp(xs, given={'flippers': 1})
    /// array([-2.34890127, -2.1883812 , -1.06062736, -0.80701174])
    /// ```
    fn logp(
        &self,
        values: &PyAny,
        given: Option<&PyDict>,
        state_ixs: Option<Vec<usize>>,
    ) -> PyResult<DataFrameLike> {
        let df_vals =
            pandas_to_logp_values(values, &self.col_indexer, &self.engine)?;

        let given = dict_to_given(given, &self.engine, &self.col_indexer)?;

        let logps = self
            .engine
            .logp(
                &df_vals.col_names,
                &df_vals.values,
                &given,
                state_ixs.as_deref(),
            )
            .map_err(to_pyerr)?;

        if logps.len() > 1 {
            Ok(DataFrameLike::Series(Series::new("logp", logps)))
        } else {
            Ok(DataFrameLike::Float(logps[0]))
        }
    }

    fn logp_scaled(
        &self,
        values: &PyAny,
        given: Option<&PyDict>,
        state_ixs: Option<Vec<usize>>,
    ) -> PyResult<DataFrameLike> {
        let df_vals =
            pandas_to_logp_values(values, &self.col_indexer, &self.engine)?;

        let given = dict_to_given(given, &self.engine, &self.col_indexer)?;

        let logps = self.engine._logp_unchecked(
            &df_vals.col_ixs.ok_or_else(|| {
                PyIndexError::new_err("Cannot compute logp on unknown columns")
            })?,
            &df_vals.values,
            &given,
            state_ixs.as_deref(),
            true,
        );

        if logps.len() > 1 {
            Ok(DataFrameLike::Series(Series::new("logp_scaled", logps)))
        } else {
            Ok(DataFrameLike::Float(logps[0]))
        }
    }

    #[pyo3(signature=(col, rows=None, values=None, state_ixs=None))]
    fn surprisal(
        &self,
        col: &PyAny,
        rows: Option<&PyAny>,
        values: Option<&PyAny>,
        state_ixs: Option<Vec<usize>>,
    ) -> PyResult<PyDataFrame> {
        let col_ix = utils::value_to_index(col, &self.col_indexer)?;
        let row_ixs: Vec<usize> = rows.map_or_else(
            || Ok((0..self.engine.shape().0).collect()),
            |vals| utils::pyany_to_indices(vals, &self.row_indexer),
        )?;

        let ftype = self.engine.ftype(col_ix).map_err(to_pyerr)?;

        if let Some(vals) = values {
            let n_vals = vals.len()?;
            match row_ixs.len().cmp(&1) {
                Ordering::Greater => {
                    let vals = if n_vals != row_ixs.len() {
                        Err(PyErr::new::<PyValueError, _>(format!(
                            "The lengths of `rows` ({}) and `values` ({}) do not match.",
                            row_ixs.len(), n_vals
                        )))
                    } else {
                        utils::pyany_to_data(vals, ftype)
                    }?;
                    let mut row_names = Vec::with_capacity(n_vals);
                    let mut surps = Vec::with_capacity(n_vals);
                    vals.iter().zip(row_ixs).try_for_each(|(x, row_ix)| {
                        // TODO: fix clone
                        self.engine
                            .surprisal(x, row_ix, col_ix, state_ixs.clone())
                            .map_err(to_pyerr)
                            .map(|surp| {
                                row_names.push(
                                    self.row_indexer.to_name[&row_ix]
                                        .to_owned(),
                                );
                                surps.push(surp);
                            })
                    })?;
                    let mut df = DataFrame::default();
                    let vals_srs = utils::vec_to_srs(
                        vals,
                        col_ix,
                        ftype,
                        &self.engine.codebook,
                    )?;
                    df.with_column(Series::new("index", row_names))
                        .map_err(to_pyerr)?;
                    df.with_column(vals_srs.0).map_err(to_pyerr)?;
                    df.with_column(Series::new("surprisal", surps))
                        .map_err(to_pyerr)?;
                    Ok(PyDataFrame(df))
                }
                Ordering::Equal => {
                    let vals = utils::pyany_to_data(vals, ftype)?;
                    let mut surps = Vec::with_capacity(n_vals);
                    let row_ix = row_ixs[0];
                    vals.iter().try_for_each(|x| {
                        // TODO: fix clone
                        self.engine
                            .surprisal(x, row_ix, col_ix, state_ixs.clone())
                            .map_err(to_pyerr)
                            .map(|surp| {
                                // row_names.push(
                                //     self.row_indexer.to_name[&row_ix].to_owned(),
                                // );
                                surps.push(surp);
                            })
                    })?;
                    let mut df = DataFrame::default();
                    df.with_column(Series::new("surprisal", surps))
                        .map_err(to_pyerr)?;
                    Ok(PyDataFrame(df))
                }
                Ordering::Less => {
                    Err(PyErr::new::<PyValueError, _>("row_ixs was empty"))
                }
            }
        } else {
            let n_rows = row_ixs.len();
            let mut row_names = Vec::with_capacity(n_rows);
            let mut surps = Vec::with_capacity(n_rows);
            let mut values = Vec::with_capacity(n_rows);
            row_ixs.iter().try_for_each(|&row_ix| {
                self.engine
                    .datum(row_ix, col_ix)
                    .map_err(to_pyerr)
                    .and_then(|x| {
                        self.engine
                            .surprisal(&x, row_ix, col_ix, state_ixs.clone())
                            .map_err(to_pyerr)
                            .map(|surp| {
                                if let Some(s) = surp {
                                    let row_name = self.row_indexer.to_name
                                        [&row_ix]
                                        .to_owned();
                                    row_names.push(row_name);
                                    values.push(x);
                                    surps.push(s);
                                }
                            })
                    })
            })?;

            let ftype = self.engine.ftype(col_ix).map_err(to_pyerr)?;
            let mut df = DataFrame::default();
            let index = Series::new("index", row_names);
            let values = utils::vec_to_srs(
                values,
                col_ix,
                ftype,
                &self.engine.codebook,
            )?;
            let surps = Series::new("surprisal", surps);

            df.with_column(index).map_err(to_pyerr)?;
            df.with_column(values.0).map_err(to_pyerr)?;
            df.with_column(surps).map_err(to_pyerr)?;
            Ok(PyDataFrame(df))
        }
    }

    #[pyo3(signature=(row, wrt=None))]
    fn novelty(&self, row: &PyAny, wrt: Option<&PyAny>) -> PyResult<f64> {
        let row_ix = utils::value_to_index(row, &self.row_indexer)?;
        let wrt = wrt
            .map(|cols| utils::pyany_to_indices(cols, &self.col_indexer))
            .transpose()?;
        self.engine
            .novelty(row_ix, wrt.as_deref())
            .map_err(|err| PyErr::new::<PyIndexError, _>(err.to_string()))
    }

    #[pyo3(signature=(cols, n_mc_samples=1000))]
    fn entropy(&self, cols: &PyAny, n_mc_samples: usize) -> PyResult<f64> {
        let col_ixs = utils::pyany_to_indices(cols, &self.col_indexer)?;
        self.engine
            .entropy(&col_ixs, n_mc_samples)
            .map_err(to_pyerr)
    }

    /// Impute (predict) the value of specific cells in the table
    ///
    /// Parameters
    /// ----------
    /// col: str or int
    ///     The column name or index to impute
    /// rows: list(str) or list(int), optional
    ///     Optional list of rows to impute. If None (default), all missing
    ///     cells will be selected.
    /// unc_type: str, optional
    ///     Can be `'js_divergence'` (default), `'pairwise_kl'` or `None`. If
    ///     None, uncertainty will not be computed.
    ///
    /// Returns
    /// -------
    /// df: polars.DataFrame
    ///     A data frame with columns for row names, values, and optional
    ///     uncertainty
    #[pyo3(signature=(col, rows=None, unc_type="js_divergence"))]
    fn impute(
        &mut self,
        col: &PyAny,
        rows: Option<&PyAny>,
        unc_type: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        use lace::cc::feature::Feature;
        use lace::ImputeUncertaintyType;

        let unc_type_opt = match unc_type {
            Some("js_divergence") => {
                Ok(Some(ImputeUncertaintyType::JsDivergence))
            }
            Some("pairwise_kl") => Ok(Some(ImputeUncertaintyType::PairwiseKl)),
            Some(val) => Err(PyErr::new::<PyValueError, _>(format!(
                "Invalid unc_type: '{val}'"
            ))),
            None => Ok(None),
        }?;

        let col_ix = utils::value_to_index(col, &self.col_indexer)?;

        let mut row_ixs: Vec<usize> = if let Some(row_ixs) = rows {
            pyany_to_indices(row_ixs, &self.row_indexer)?
        } else {
            // Get all missing rows
            let ftr = self.engine.states[0].feature(col_ix);
            (0..self.engine.shape().0)
                .filter(|&ix| ftr.is_missing(ix))
                .collect()
        };

        let mut values = Vec::with_capacity(row_ixs.len());
        let mut uncs = Vec::with_capacity(row_ixs.len());
        let mut row_names = Vec::with_capacity(row_ixs.len());

        row_ixs.drain(..).try_for_each(|row_ix| {
            self.engine
                .impute(row_ix, col_ix, unc_type_opt)
                .map(|(val, unc)| {
                    values.push(val);
                    row_names.push(self.row_indexer.to_name[&row_ix].clone());
                    if let Some(u) = unc {
                        uncs.push(u)
                    };
                })
                .map_err(to_pyerr)
        })?;

        let ftype = self.engine.ftype(col_ix).map_err(to_pyerr)?;

        let df = {
            let mut df = DataFrame::default();
            let values_srs = utils::vec_to_srs(
                values,
                col_ix,
                ftype,
                &self.engine.codebook,
            )?;
            let index = Series::new("index", row_names);
            df.with_column(index).map_err(to_pyerr)?;
            df.with_column(values_srs.0).map_err(to_pyerr)?;

            if !uncs.is_empty() {
                let uncs_srs = Series::new("uncertainty", uncs);
                df.with_column(uncs_srs).map_err(to_pyerr)?;
            }
            df
        };

        Ok(PyDataFrame(df))
    }

    /// Predict a single target from a conditional distribution
    ///
    /// Parameters
    /// ----------
    /// target: int or str
    ///     The column to predict
    /// given: dict, optional
    ///     Column -> Value dictionary describing observations. Note that
    ///     columns can either be indices (int) or names (str)
    /// state_ixs: list[int], optionals, optionals
    //     A list of indices of states to use for prediction. If `None`
    //     (default), use all states.
    /// with_uncertainty: bool. optional
    ///     if `True` (default), return the uncertainty
    ///
    /// Returns
    /// -------
    /// pred: value
    ///     The predicted value
    /// unc: float, optional
    ///     The uncertainty
    #[pyo3(signature=(target, given=None, state_ixs=None, with_uncertainty=true))]
    fn predict(
        &self,
        target: &PyAny,
        given: Option<&PyDict>,
        state_ixs: Option<Vec<usize>>,
        with_uncertainty: bool,
    ) -> PyResult<Py<PyAny>> {
        let col_ix = value_to_index(target, &self.col_indexer)?;
        let given = dict_to_given(given, &self.engine, &self.col_indexer)?;

        if with_uncertainty {
            let unc_type_opt = Some(PredictUncertaintyType::JsDivergence);
            let (pred, unc) = self
                .engine
                .predict(col_ix, &given, unc_type_opt, state_ixs.as_deref())
                .map_err(|err| {
                    PyErr::new::<PyValueError, _>(format!("{err}"))
                })?;
            let value = datum_to_value(pred)?;
            Python::with_gil(|py| {
                let unc = unc.into_py(py);
                Ok((value, unc).into_py(py))
            })
        } else {
            let (pred, _) = self
                .engine
                .predict(col_ix, &given, None, state_ixs.as_deref())
                .map_err(|err| {
                    PyErr::new::<PyValueError, _>(format!("{err}"))
                })?;
            datum_to_value(pred)
        }
    }

    /// Forward the Markov chains
    ///
    /// Parameters
    /// ----------
    /// n_iters: int
    ///     Number of iterations/steps to run the Markov chains
    /// checkpoint: int, optional
    ///     The number of iterations/steps between saving off the metadata
    ///     (save_path must be provided to save)
    /// timeout: int, optional
    ///     The max number of seconds to update. If None (default), will run
    ///     for `n_iters` iterations.
    /// transitions: list(str), optional
    ///     List of the transitions to run. If None (default), the default set
    ///     is run.
    /// save_path: pathlike, optional
    ///     Where to save the metadata on checkpoints and when the run is
    ///     complete.
    /// quiet: bool, optional
    ///     If True, do not display progress.
    ///
    /// Examples
    /// --------
    ///
    /// Run for 100 steps
    ///
    /// >>> import lace
    /// >>>
    /// >>> engine = lace.Engine('animals.rp')
    /// >>> engine.update(100)
    ///
    /// Run 100 iterations with the `sams` (split mere) row reassignment
    /// algorithm and the `gibbs` column reassignment algorithm.
    ///
    /// >>> transitions = [
    /// ...     "state_alpha",              # CRP alpha for states
    /// ...     "view_alphas",              # CRP alpha for views
    /// ...     "feature_priors",           # column prior params
    /// ...     "row_assignment(sams)"      # row reassignment w/ sams kernel
    /// ...     "component_params",         # component (category) params
    /// ...     "column_assignment(gibbs)"  # col reassignment w/ gibbs kernel
    /// ...     "component_params",         # component (category) params
    /// ... ]
    /// >>>
    /// >>> engine.update(
    /// ...     100,
    /// ...     checkpoint=25,
    /// ...     timeout=60,
    /// ...     transitions=transitions,
    /// ...     save_path='animals-updated.lace',
    /// ... )
    #[pyo3(
        signature = (
            n_iters,
            timeout=None,
            checkpoint=None,
            transitions=None,
            save_path=None,
            quiet=false,
        )
    )]
    fn update(
        &mut self,
        n_iters: usize,
        timeout: Option<u64>,
        checkpoint: Option<usize>,
        transitions: Option<Vec<transition::StateTransition>>,
        save_path: Option<PathBuf>,
        quiet: bool,
    ) {
        use lace::update_handler::{ProgressBar, Timeout};
        use std::time::Duration;

        let config = match transitions {
            Some(mut trns) => {
                let trns = trns.drain(..).map(|t| t.into()).collect();
                EngineUpdateConfig::new().transitions(trns)
            }
            None => EngineUpdateConfig::with_default_transitions(),
        }
        .n_iters(n_iters)
        .checkpoint(checkpoint);

        let save_config =
            save_path.map(|path| lace::config::SaveEngineConfig {
                path,
                ser_type: SerializedType::Bincode,
            });

        let config = EngineUpdateConfig {
            save_config,
            ..config
        };

        let timeout = {
            let secs = timeout.unwrap_or(std::u64::MAX);
            Timeout::new(Duration::from_secs(secs))
        };

        if quiet {
            self.engine.update(config, timeout).unwrap();
        } else {
            let pbar = ProgressBar::new();
            self.engine.update(config, (timeout, pbar)).unwrap();
        }
    }

    /// Append new rows to the table.
    ///
    /// Parameters
    /// ----------
    /// rows: pandas.DataFrame or pandas.Series
    ///     The rows to append. If rows is a series, the series will be
    ///     considered one row.
    ///
    /// Example
    ///
    /// >>> import pandas as pd
    /// >>> import lace
    /// >>>
    /// >>> engine = lace.Engine('animals.rp')
    /// >>>
    /// >>> row = pd.Series(
    /// ...     [0, 1, 0],
    /// ...     name='tribble',
    /// ...     index=['hunter', 'fierce', 'meatteeth'],
    /// ... )
    /// >>>
    /// >>> engine.append_rows(row)
    fn append_rows(&mut self, rows: &PyAny) -> PyResult<()> {
        let df_vals = pandas_to_insert_values(
            rows,
            &self.col_indexer,
            &self.engine,
            None,
        )
        .and_then(|df_vals| {
            if df_vals.row_names.is_none() {
                Err(PyErr::new::<PyValueError, _>(
                    "Values must have index to provide row names. For polars, \
                    an 'index' column must be added",
                ))
            } else {
                Ok(df_vals)
            }
        })?;

        // must add new row names to indexer
        let row_names = df_vals.row_names.unwrap();
        (self.engine.n_rows()..).zip(row_names.iter()).for_each(
            |(ix, name)| {
                // row names passed to 'append' should not exist
                assert!(!self.row_indexer.to_ix.contains_key(name));
                self.row_indexer.to_ix.insert(name.to_owned(), ix);
                self.row_indexer.to_name.insert(ix, name.to_owned());
            },
        );

        let data = parts_to_insert_values(
            df_vals.col_names,
            row_names,
            df_vals.values,
        );

        // TODO: Return insert ops
        let write_mode = lace::WriteMode {
            overwrite: lace::OverwriteMode::Deny,
            insert: lace::InsertMode::DenyNewColumns,
            ..Default::default()
        };
        self.engine
            .insert_data(data, None, None, write_mode)
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{err}")))?;

        Ok(())
    }

    /// Append new columns to the Engine
    fn append_columns(
        &mut self,
        cols: &PyAny,
        mut metadata: Vec<metadata::ColumnMetadata>,
    ) -> PyResult<()> {
        let suppl_types = Some(
            metadata
                .iter()
                .map(|md| (md.0.name.clone(), coltype_to_ftype(&md.0.coltype)))
                .collect::<HashMap<String, FType>>(),
        );

        let df_vals = pandas_to_insert_values(
            cols,
            &self.col_indexer,
            &self.engine,
            suppl_types.as_ref(),
        )
        .and_then(|df_vals| {
            if df_vals.row_names.is_none() {
                Err(PyErr::new::<PyValueError, _>(
                    "Values must have index to provide row names. For polars, \
                    an 'index' column must be added",
                ))
            } else {
                Ok(df_vals)
            }
        })?;

        let write_mode = lace::WriteMode {
            overwrite: lace::OverwriteMode::Deny,
            insert: lace::InsertMode::DenyNewRows,
            ..Default::default()
        };

        // must add new col names to indexer
        let col_names = df_vals.col_names;
        (self.engine.n_cols()..).zip(col_names.iter()).for_each(
            |(ix, name)| {
                // columns names passed to 'append' should not exist
                assert!(!self.col_indexer.to_ix.contains_key(name));
                self.col_indexer.to_ix.insert(name.to_owned(), ix);
                self.col_indexer.to_name.insert(ix, name.to_owned());
            },
        );

        let data = parts_to_insert_values(
            col_names,
            df_vals.row_names.unwrap(),
            df_vals.values,
        );

        let col_metadata = ColMetadataList::try_from(
            metadata.drain(..).map(|md| md.0).collect::<Vec<_>>(),
        )
        .map_err(to_pyerr)?;

        self.engine
            .insert_data(data, Some(col_metadata), None, write_mode)
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{err}")))?;

        Ok(())
    }

    /// Delete a given column from the ``Engine``
    fn del_column(&mut self, col: &PyAny) -> PyResult<()> {
        let col_ix = utils::value_to_index(col, &self.col_indexer)?;
        self.col_indexer.drop_by_ix(col_ix)?;
        self.engine.del_column(col_ix).map_err(to_pyerr)
    }

    /// Edit the datum in a cell in the PCC table
    ///
    /// Parameters
    /// ----------
    /// row: row index
    ///     The row index of the cell to edit
    /// col: column index
    ///     The column index of the cell to edit
    /// value: value
    ///     The new value at the cell
    fn edit_cell(
        &mut self,
        row: &PyAny,
        col: &PyAny,
        value: &PyAny,
    ) -> PyResult<()> {
        let row_ix = utils::value_to_index(row, &self.row_indexer)?;
        let col_ix = utils::value_to_index(col, &self.col_indexer)?;
        let datum = {
            let ftype = self.engine.ftype(col_ix).map_err(to_pyerr)?;
            utils::value_to_datum(value, ftype)?
        };
        let row = lace::Row {
            row_ix,
            values: vec![lace::Value {
                col_ix,
                value: datum,
            }],
        };
        let write_mode = lace::WriteMode::unrestricted();
        self.engine
            .insert_data(vec![row], None, None, write_mode)
            .map_err(to_pyerr)?;
        Ok(())
    }

    fn categorical_support(
        &self,
        col: &PyAny,
    ) -> PyResult<Vec<pyo3::Py<PyAny>>> {
        use lace::codebook::ValueMap as Vm;
        let col_ix = utils::value_to_index(col, &self.col_indexer)?;
        Python::with_gil(|py| {
            self.engine
                .codebook
                .value_map(col_ix)
                .ok_or_else(|| {
                    let msg = format!("No value map for column {col_ix}");
                    PyIndexError::new_err(msg)
                })
                .map(|vm| match vm {
                    Vm::U8(k) => (0..*k as u64)
                        .map(|ix| ix.into_py(py))
                        .collect::<Vec<_>>(),
                    Vm::Bool => vec![false.into_py(py), true.into_py(py)],
                    Vm::String(cm) => (0..cm.len())
                        .map(|ix| cm.category(ix).into_py(py))
                        .collect::<Vec<_>>(),
                })
        })
    }
}

#[pyfunction]
pub fn infer_srs_metadata(
    srs: PySeries,
    cat_cutoff: u8,
    no_hypers: bool,
) -> PyResult<metadata::ColumnMetadata> {
    lace::codebook::data::series_to_colmd(&srs.0, Some(cat_cutoff), no_hypers)
        .map_err(to_pyerr)
        .map(metadata::ColumnMetadata)
}

/// A Python module implemented in Rust.
#[pymodule]
fn core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CoreEngine>()?;
    m.add_class::<CodebookBuilder>()?;
    m.add_class::<transition::ColumnKernel>()?;
    m.add_class::<transition::RowKernel>()?;
    m.add_class::<transition::StateTransition>()?;
    m.add_class::<metadata::ColumnMetadata>()?;
    m.add_class::<metadata::ValueMap>()?;
    m.add_class::<metadata::ContinuousHyper>()?;
    m.add_class::<metadata::ContinuousPrior>()?;
    m.add_class::<metadata::CategoricalHyper>()?;
    m.add_class::<metadata::CategoricalPrior>()?;
    m.add_class::<metadata::CountHyper>()?;
    m.add_class::<metadata::CountPrior>()?;
    m.add_function(wrap_pyfunction!(infer_srs_metadata, m)?)?;
    Ok(())
}

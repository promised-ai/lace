mod df;
mod utils;

use std::collections::HashMap;
use std::path::PathBuf;

use df::{DataFrameLike, PyDataFrame, PySeries};
use lace::codebook::Codebook;
use lace::data::DataSource;
use lace::{EngineUpdateConfig, HasStates, OracleT, PredictUncertaintyType};
use polars::prelude::{DataFrame, NamedFrom, Series};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::utils::*;

#[pyclass(subclass)]
struct CoreEngine {
    engine: lace::Engine,
    col_indexer: Indexer,
    row_indexer: Indexer,
    value_maps: HashMap<usize, HashMap<String, usize>>,
    rng: Xoshiro256Plus,
}

#[pyclass]
struct ColumnMaximumLogpCache(lace::ColumnMaximumLogpCache);

#[pymethods]
impl ColumnMaximumLogpCache {
    #[staticmethod]
    fn from_oracle(
        engine: &CoreEngine,
        columns: &PyAny,
        given: Option<&PyDict>,
    ) -> PyResult<Self> {
        let col_ixs = pyany_to_indices(columns, &engine.col_indexer)?;

        let given = dict_to_given(
            given,
            &engine.engine,
            &engine.col_indexer,
            &engine.value_maps,
        )?;

        let cache = lace::ColumnMaximumLogpCache::from_oracle(
            &engine.engine,
            &col_ixs,
            &given,
            None,
        );

        Ok(Self(cache))
    }
}

fn infer_src_from_extension(path: PathBuf) -> Option<DataSource> {
    path.extension()
        .map(|s| s.to_string_lossy().to_lowercase())
        .and_then(|ext| match ext.as_str() {
            "csv" | "csv.gz" => Some(DataSource::Csv(path)),
            "feather" | "ipc" => Some(DataSource::Ipc(path)),
            "parquet" => Some(DataSource::Parquet(path)),
            "json" | "jsonl" => Some(DataSource::Json(path)),
            _ => None,
        })
}

fn data_to_src(
    path: PathBuf,
    source_type: Option<&str>,
) -> Result<DataSource, String> {
    match source_type {
        Some("csv") | Some("csv.gz") => Ok(DataSource::Csv(path)),
        Some("feather") | Some("ipc") => Ok(DataSource::Ipc(path)),
        Some("parquet") => Ok(DataSource::Parquet(path)),
        Some("json") | Some("jsonl") => Ok(DataSource::Json(path)),
        Some(t) => Err(format!("Invalid source_type: `{:}`", t)),
        None => infer_src_from_extension(path.clone()).ok_or_else(|| {
            format!("Could not infer source data format. Invalid extension for {:?}", path)
        }),
    }
}

fn get_or_create_codebook(
    codebook: Option<PathBuf>,
    data_source: DataSource,
    cat_cutoff: Option<u8>,
    no_hypers: bool,
) -> Result<Codebook, String> {
    use lace::codebook::data;
    if let Some(path) = codebook {
        let file = std::fs::File::open(path.clone())
            .map_err(|err| format!("Error opening {:?}: {:}", path, err))?;

        serde_yaml::from_reader(&file)
            .or_else(|_| serde_json::from_reader(&file))
            .map_err(|_| format!("Failed to read codebook at {:?}", path))
    } else {
        let df = match data_source {
            DataSource::Csv(path) => data::read_csv(path),
            DataSource::Ipc(path) => data::read_ipc(path),
            DataSource::Json(path) => data::read_json(path),
            DataSource::Parquet(path) => data::read_parquet(path),
            _ => panic!("Empty data source not supporteed"),
        }
        .map_err(|err| format!("Failed to create codebook: {:}", err))?;
        data::df_to_codebook(&df, cat_cutoff, None, no_hypers).map_err(|err| {
            format!("Failed to create codebook from DataFrame: {:}", err)
        })
    }
}

// FIXME: implement __repr__
// FIXME: implement name (get name from codebook)
#[pymethods]
impl CoreEngine {
    /// Load a Engine from metadata
    #[classmethod]
    fn load(_cls: &PyType, path: PathBuf) -> CoreEngine {
        let engine = lace::Engine::load(path).unwrap();
        CoreEngine {
            col_indexer: Indexer::columns(&engine.codebook),
            row_indexer: Indexer::rows(&engine.codebook),
            value_maps: value_maps(&engine.codebook),
            rng: Xoshiro256Plus::from_entropy(),
            engine,
        }
    }

    /// Create a new Engine from the prior
    ///
    /// Parameters
    /// ----------
    /// data_source: Pathlike
    ///     The path to the data file
    /// codebook: Pathlike, optional
    ///     The path to the codebook. If None (default), the default codebook is
    ///     generated and used.
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
            data_source,
            codebook=None,
            n_states=16,
            id_offset=0,
            rng_seed=None,
            source_type=None,
            cat_cutoff=20,
            no_hypers=false
        )
    )]
    fn new(
        data_source: PathBuf,
        codebook: Option<PathBuf>,
        n_states: usize,
        id_offset: usize,
        rng_seed: Option<u64>,
        source_type: Option<&str>,
        cat_cutoff: Option<u8>,
        no_hypers: bool,
    ) -> Result<CoreEngine, PyErr> {
        let data_source = data_to_src(data_source, source_type)
            .map_err(PyErr::new::<PyValueError, _>)?;
        let codebook = get_or_create_codebook(
            codebook,
            data_source.clone(),
            cat_cutoff,
            no_hypers,
        )
        .map_err(PyErr::new::<PyValueError, _>)?;
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
            value_maps: value_maps(&engine.codebook),
            rng,
            engine,
        })
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
        self.engine
            .codebook
            .row_names
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn __getitem__(&self, ix: &PyAny) -> PyResult<PySeries> {
        let col_ix = utils::value_to_index(ix, &self.col_indexer)?;

        let values: Vec<lace::Datum> = (0..self.engine.shape().0)
            .map(|row_ix| self.engine.datum(row_ix, col_ix).map_err(to_pyerr))
            .collect::<PyResult<Vec<_>>>()?;

        let ftype = self.engine.ftype(col_ix).map_err(to_pyerr)?;

        utils::vec_to_srs(values, col_ix, ftype, &self.engine.codebook)
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
        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        )?;

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
    ) -> PyResult<DataFrameLike> {
        let df_vals = pandas_to_logp_values(
            values,
            &self.col_indexer,
            &self.engine,
            &self.value_maps,
        )?;

        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        )?;

        let logps = self
            .engine
            .logp(&df_vals.col_ixs, &df_vals.values, &given, None)
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
        col_max_logps: Option<&ColumnMaximumLogpCache>,
    ) -> PyResult<DataFrameLike> {
        let df_vals = pandas_to_logp_values(
            values,
            &self.col_indexer,
            &self.engine,
            &self.value_maps,
        )?;

        let cache_opt = if col_max_logps.is_none() {
            Python::with_gil(|py| {
                let obj = df_vals.col_ixs.clone().into_py(py);
                let cols: &PyList = obj.downcast(py).unwrap();
                Some(
                    ColumnMaximumLogpCache::from_oracle(self, cols, given)
                        .map_err(to_pyerr),
                )
            })
            .transpose()?
        } else {
            None
        };

        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        )?;

        let logps = self.engine._logp_unchecked(
            &df_vals.col_ixs,
            &df_vals.values,
            &given,
            None,
            col_max_logps.or(cache_opt.as_ref()).map(|cache| &cache.0),
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
            let vals = if n_vals != row_ixs.len() {
                Err(PyErr::new::<PyValueError, _>(format!(
                    "The lengths of `rows` ({}) and `values` ({}) do not match.",
                    row_ixs.len(), n_vals
                )))
            } else {
                utils::pyany_to_data(vals, col_ix, ftype, &self.value_maps)
            }?;
            let mut row_names = Vec::with_capacity(n_vals);
            let mut surps = Vec::with_capacity(n_vals);
            vals.iter().zip(row_ixs).try_for_each(|(x, row_ix)| {
                // TODO: fix clone
                self.engine
                    .surprisal(x, row_ix, col_ix, state_ixs.clone())
                    .map_err(to_pyerr)
                    .map(|surp| {
                        row_names
                            .push(self.row_indexer.to_name[&row_ix].to_owned());
                        surps.push(surp);
                    })
            })?;
            let mut df = DataFrame::default();
            let vals_srs =
                utils::vec_to_srs(vals, col_ix, ftype, &self.engine.codebook)?;
            df.with_column(Series::new("index", row_names))
                .map_err(to_pyerr)?;
            df.with_column(vals_srs.0).map_err(to_pyerr)?;
            df.with_column(Series::new("surprisal", surps))
                .map_err(to_pyerr)?;
            Ok(PyDataFrame(df))
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
    /// with_uncertainty: bool. optional
    ///     if `True` (default), return the uncertainty
    ///
    /// Returns
    /// -------
    /// pred: value
    ///     The predicted value
    /// unc: float, optional
    ///     The uncertainty
    #[pyo3(signature=(target, given=None, with_uncertainty=true))]
    fn predict(
        &self,
        target: &PyAny,
        given: Option<&PyDict>,
        with_uncertainty: bool,
    ) -> PyResult<Py<PyAny>> {
        let col_ix = value_to_index(target, &self.col_indexer)?;
        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        )?;

        if with_uncertainty {
            let unc_type_opt = Some(PredictUncertaintyType::JsDivergence);
            let (pred, unc) = self
                .engine
                .predict(col_ix, &given, unc_type_opt, None)
                .map_err(|err| {
                    PyErr::new::<PyValueError, _>(format!("{err}"))
                })?;
            let value = datum_to_value(pred, col_ix, &self.engine.codebook)?;
            Python::with_gil(|py| {
                let unc = unc.into_py(py);
                Ok((value, unc).into_py(py))
            })
        } else {
            let (pred, _) =
                self.engine.predict(col_ix, &given, None, None).map_err(
                    |err| PyErr::new::<PyValueError, _>(format!("{err}")),
                )?;
            datum_to_value(pred, col_ix, &self.engine.codebook)
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
        )
    )]
    fn update(
        &mut self,
        n_iters: usize,
        timeout: Option<u64>,
        checkpoint: Option<usize>,
        transitions: Option<Vec<String>>,
        save_path: Option<PathBuf>,
    ) {
        let config = match transitions {
            Some(ref trns) => {
                EngineUpdateConfig::new().transitions(parse_transitions(trns))
            }
            None => EngineUpdateConfig::with_default_transitions(),
        }
        .n_iters(n_iters)
        .timeout(timeout)
        .checkpoint(checkpoint);

        let save_config = save_path.map(|path| {
            let file_config = lace::metadata::FileConfig {
                metadata_version: lace::metadata::latest::METADATA_VERSION,
                serialized_type: lace::metadata::SerializedType::Bincode,
            };
            lace::config::SaveEngineConfig { path, file_config }
        });

        let config = EngineUpdateConfig {
            save_config,
            ..config
        };

        self.engine.update(config, None, None).unwrap();
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
            &self.value_maps,
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

        let data =
            parts_to_insert_values(df_vals.col_ixs, row_names, df_vals.values);

        // TODO: Return insert ops
        self.engine
            .insert_data(data, None, None, lace::WriteMode::unrestricted())
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{err}")))?;

        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "lace_core")]
fn lace_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ColumnMaximumLogpCache>()?;
    m.add_class::<CoreEngine>()?;
    Ok(())
}

mod df;
mod utils;

use std::collections::HashMap;
use std::path::PathBuf;

use braid::codebook::Codebook;
use braid::data::DataSource;
use braid::{EngineUpdateConfig, HasStates, OracleT, PredictUncertaintyType};
use df::{PyDataFrame, PySeries};
use polars::prelude::{NamedFrom, Series};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::utils::*;

#[pyclass]
struct Engine {
    engine: braid::Engine,
    col_indexer: Indexer,
    row_indexer: Indexer,
    value_maps: HashMap<usize, HashMap<String, usize>>,
    rng: Xoshiro256Plus,
}

#[pyclass]
struct ColumnMaximumLogpCache(braid::ColumnMaximumLogpCache);

#[pymethods]
impl ColumnMaximumLogpCache {
    #[staticmethod]
    fn from_oracle(
        engine: &Engine,
        columns: &PyList,
        given: Option<&PyDict>,
    ) -> PyResult<Self> {
        let col_ixs = column_indices(columns, &engine.col_indexer)?;

        let given = dict_to_given(
            given,
            &engine.engine,
            &engine.col_indexer,
            &engine.value_maps,
        )?;

        let cache = braid::ColumnMaximumLogpCache::from_oracle(
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
    use braid::codebook::data;
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

#[pymethods]
impl Engine {
    /// Load a Engine from metadata
    #[staticmethod]
    fn load(path: PathBuf) -> Engine {
        let engine = braid::Engine::load(path, None).unwrap();
        Engine {
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
    ) -> Result<Engine, PyErr> {
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

        let engine = braid::Engine::new(
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

    fn n_states(&self) -> usize {
        self.engine.n_states()
    }

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
    /// >>> import pybraid
    /// >>> engine = pybraid.Engine('animals.rp')
    /// >>> engine.depprob([(0, 1), (1, 2)])
    /// array([0.125, 0.   ])
    /// >>> engine.depprob([('swims', 'flippers'), ('swims', 'fast')])
    /// array([0.875, 0.25 ])
    fn depprob(&self, col_pairs: &PyList) -> PyResult<PySeries> {
        let pairs = list_to_pairs(col_pairs, &self.col_indexer)?;
        self.engine
            .depprob_pw(&pairs)
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{}", err)))
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
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{}", err)))
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
    /// >>> import pybraid
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
        wrt: Option<&PyList>,
        col_weighted: bool,
    ) -> PyResult<PySeries> {
        let variant = if col_weighted {
            braid::RowSimilarityVariant::ColumnWeighted
        } else {
            braid::RowSimilarityVariant::ViewWeighted
        };

        let pairs = list_to_pairs(row_pairs, &self.row_indexer)?;
        if let Some(cols) = wrt {
            let wrt = column_indices(cols, &self.col_indexer)?;
            self.engine.rowsim_pw(&pairs, Some(wrt.as_slice()), variant)
        } else {
            self.engine.rowsim_pw::<_, usize>(&pairs, None, variant)
        }
        .map_err(|err| PyErr::new::<PyValueError, _>(format!("{}", err)))
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
        cols: &PyList,
        given: Option<&PyDict>,
        n: usize,
    ) -> PyResult<PyDataFrame> {
        let col_ixs = column_indices(cols, &self.col_indexer)?;
        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        )?;

        let values = self
            .engine
            .simulate(&col_ixs, &given, n, None, &mut self.rng)
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;

        utils::simulate_to_df(
            values,
            &self.engine.ftypes(),
            &col_ixs,
            &self.col_indexer,
            &self.engine.codebook,
        )
    }

    /// Compute the log likelihood of data given optional conditions
    ///
    /// Parameters
    /// ----------
    /// values: pandas.DataFrame or pandas.Series
    ///     The input data. Each row is a record. Column names must match column
    ///     names in the braid table.
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
    /// >>> import pybraid
    /// >>>
    /// >>> engine = pybraid.Engine('animals.rp')
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
    ) -> PyResult<PySeries> {
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
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{err}")))?;

        Ok(PySeries(Series::new("logp", logps)))
    }

    fn logp_scaled(
        &self,
        values: &PyAny,
        given: Option<&PyDict>,
        col_max_logps: Option<&ColumnMaximumLogpCache>,
    ) -> PyResult<PySeries> {
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

        let logps = self.engine._logp_unchecked(
            &df_vals.col_ixs,
            &df_vals.values,
            &given,
            None,
            col_max_logps.map(|cache| &cache.0),
        );

        Ok(PySeries(Series::new("logp", logps)))
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
    /// >>> import pybraid
    /// >>>
    /// >>> engine = pybraid.Engine('animals.rp')
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
    /// ...     save_path='animals-updated.rp',
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
        transitions: Option<Vec<String>>,
        save_path: Option<PathBuf>,
        quiet: bool,
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
            let save_config = braid::metadata::SaveConfig {
                metadata_version: braid::metadata::latest::METADATA_VERSION,
                serialized_type: braid::metadata::SerializedType::Bincode,
                user_info: braid::metadata::UserInfo {
                    encryption_key: None,
                    profile: None,
                },
            };
            braid::config::SaveEngineConfig { path, save_config }
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
    /// >>> import pybraid
    /// >>>
    /// >>> engine = pybraid.Engine('animals.rp')
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
            .insert_data(data, None, None, braid::WriteMode::unrestricted())
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("{err}")))?;

        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "pybraid")]
fn pybraid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ColumnMaximumLogpCache>()?;
    m.add_class::<Engine>()?;
    Ok(())
}

mod utils;

use std::collections::HashMap;
use std::path::PathBuf;

use braid::{EngineUpdateConfig, OracleT, HasStates, PredictUncertaintyType};
use numpy::{IntoPyArray, PyArray1, ToPyArray};
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
struct ColumnMaximumLogpCache(
    braid::ColumnMaximumLogpCache
);

#[pymethods]
impl ColumnMaximumLogpCache {
    #[staticmethod]
    fn from_oracle(
        engine: &Engine,
        columns: &PyList,
        given: Option<&PyDict>,
    ) -> Self {
        let col_ixs = column_indices(columns, &engine.col_indexer);

        let given = dict_to_given(
            given,
            &engine.engine,
            &engine.col_indexer,
            &engine.value_maps,
        );

        let cache = braid::ColumnMaximumLogpCache::from_oracle(
            &engine.engine,
            &col_ixs,
            &given,
            None,
        );

        Self(cache)
    }

}

#[pymethods]
impl Engine {
    /// Create a Engine from metadata
    #[new]
    fn new(path: PathBuf) -> Engine {
        let engine = braid::Engine::load(path, None).unwrap();
        Engine {
            col_indexer: Indexer::columns(&engine.codebook),
            row_indexer: Indexer::rows(&engine.codebook),
            value_maps: value_maps(&engine.codebook),
            rng: Xoshiro256Plus::from_entropy(),
            engine,
        }
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
    fn depprob<'py>(
        &self,
        py: Python<'py>,
        col_pairs: &PyList,
    ) -> &'py PyArray1<f64> {
        let pairs = list_to_pairs(col_pairs, &self.col_indexer);
        self.engine.depprob_pw(&pairs).unwrap().to_pyarray(py)
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
    #[args(col_weighted = "false")]
    fn rowsim<'py>(
        &self,
        py: Python<'py>,
        row_pairs: &PyList,
        wrt: Option<&PyList>,
        col_weighted: bool,
    ) -> &'py PyArray1<f64> {
        let pairs = list_to_pairs(row_pairs, &self.row_indexer);
        if let Some(cols) = wrt {
            let wrt = column_indices(cols, &self.col_indexer);
            self.engine
                .rowsim_pw(&pairs, Some(wrt.as_slice()), col_weighted)
        } else {
            self.engine.rowsim_pw(&pairs, None, col_weighted)
        }
        .unwrap()
        .to_pyarray(py)
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
    #[args(n = "1")]
    fn simulate(
        &mut self,
        cols: &PyList,
        given: Option<&PyDict>,
        n: usize,
    ) -> Py<PyAny> {
        let col_ixs = column_indices(cols, &self.col_indexer);
        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        );

        let mut data = self
            .engine
            .simulate(&col_ixs, &given, n, None, &mut self.rng)
            .unwrap();

        Python::with_gil(|py| {
            data.drain(..)
                .map(|mut row| {
                    row.drain(..)
                        .zip(col_ixs.iter())
                        .map(|(datum, &ix)| {
                            datum_to_value(datum, ix, &self.engine.codebook)
                        })
                        .collect::<Vec<_>>()
                        .into_py(py)
                })
                .collect::<Vec<_>>()
                .into_py(py)
        })
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
    fn logp<'py>(
        &self,
        py: Python<'py>,
        values: &PyAny,
        given: Option<&PyDict>,
    ) -> &'py PyArray1<f64> {
        let (col_ixs, _, data) = pandas_to_logp_values(
            values,
            &self.col_indexer,
            &self.engine,
            &self.value_maps,
        );

        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        );

        let logps = self
            .engine
            .logp(&col_ixs, &data, &given, None)
            .unwrap();

        logps.into_pyarray(py)
    }

    fn logp_scaled<'py>(
        &self,
        py: Python<'py>,
        values: &PyAny,
        given: Option<&PyDict>,
        col_max_logps: Option<&ColumnMaximumLogpCache>,
    ) -> &'py PyArray1<f64> {
        let (col_ixs, _, data) = pandas_to_logp_values(
            values,
            &self.col_indexer,
            &self.engine,
            &self.value_maps,
        );

        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        );

        let logps = self.engine.logp_unchecked(
            &col_ixs,
            &data,
            &given,
            None, 
            col_max_logps.map(|cache| &cache.0),
        );

        logps.into_pyarray(py)
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
    #[args(with_uncertainty = "true")]
    fn predict(
        &self,
        target: &PyAny,
        given: Option<&PyDict>,
        with_uncertainty: bool,
    ) -> Py<PyAny> {
        let col_ix = value_to_index(target, &self.col_indexer);
        let given = dict_to_given(
            given,
            &self.engine,
            &self.col_indexer,
            &self.value_maps,
        );

        if with_uncertainty {
            let unc_type_opt = Some(PredictUncertaintyType::JsDivergence);
            let (pred, unc) =
                self.engine.predict(col_ix, &given, unc_type_opt, None).unwrap();
            let value = datum_to_value(pred, col_ix, &self.engine.codebook);
            Python::with_gil(|py| {
                let unc = unc.into_py(py);
                (value, unc).into_py(py)
            })
        } else {
            let (pred, _) = self.engine.predict(col_ix, &given, None, None).unwrap();
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
    fn append_rows(&mut self, rows: &PyAny) {
        let (col_ixs, row_names, data) = pandas_to_insert_values(
            rows,
            &self.col_indexer,
            &self.engine,
            &self.value_maps,
        );

        // must add new row names to indexer
        (self.engine.n_rows()..).zip(row_names.iter()).for_each(
            |(ix, name)| {
                // row names passed to 'append' should not exist
                assert!(!self.row_indexer.to_ix.contains_key(name));
                self.row_indexer.to_ix.insert(name.to_owned(), ix);
                self.row_indexer.to_name.insert(ix, name.to_owned());
            },
        );

        let data = parts_to_insert_values(col_ixs, row_names, data);

        self.engine
            .insert_data(data, None, None, braid::WriteMode::unrestricted())
            .unwrap();
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

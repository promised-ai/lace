mod utils;

use std::collections::HashMap;
use std::path::PathBuf;

use braid::{Datum, OracleT, PredictUncertaintyType};
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

#[pymethods]
impl Engine {
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

    fn depprob(&self, col_pairs: &PyList) -> Vec<f64> {
        let pairs = list_to_pairs(col_pairs, &self.col_indexer);
        self.engine.depprob_pw(&pairs).unwrap()
    }

    #[args(col_weighted = "false")]
    fn rowsim(&self, row_pairs: &PyList, wrt: Option<&PyList>, col_weighted: bool) -> Vec<f64> {
        let pairs = list_to_pairs(row_pairs, &self.row_indexer);
        if let Some(cols) = wrt {
            let wrt = column_indices(cols, &self.col_indexer);
            self.engine
                .rowsim_pw(&pairs, Some(wrt.as_slice()), col_weighted)
        } else {
            self.engine.rowsim_pw(&pairs, None, col_weighted)
        }
        .unwrap()
    }

    #[args(n = "1")]
    fn simulate(
        &mut self,
        cols: &PyList,
        given: Option<&PyDict>,
        n: usize,
        state_ixs_opt: Option<Vec<usize>>,
    ) -> Py<PyAny> {
        let col_ixs = column_indices(cols, &self.col_indexer);
        let given = dict_to_given(given, &self.engine, &self.col_indexer, &self.value_maps);

        let mut data = self
            .engine
            .simulate(&col_ixs, &given, n, state_ixs_opt, &mut self.rng)
            .unwrap();

        Python::with_gil(|py| {
            data.drain(..)
                .map(|mut row| {
                    row.drain(..)
                        .zip(col_ixs.iter())
                        .map(|(datum, &ix)| datum_to_value(datum, ix, &self.engine.codebook))
                        .collect::<Vec<_>>()
                        .into_py(py)
                })
                .collect::<Vec<_>>()
                .into_py(py)
        })
    }

    fn logp(
        &self,
        cols: &PyList,
        values: &PyList,
        given: Option<&PyDict>,
        state_ixs_opt: Option<Vec<usize>>,
    ) -> Vec<f64> {
        let col_ixs = column_indices(cols, &self.col_indexer);
        let given = dict_to_given(given, &self.engine, &self.col_indexer, &self.value_maps);

        let data: Vec<Vec<Datum>> = values
            .iter()
            .map(|row_any| {
                let row: &PyList = row_any.downcast().unwrap();
                col_ixs
                    .iter()
                    .zip(row.iter())
                    .map(|(&ix, val)| {
                        value_to_datum(val, ix, self.engine.ftype(ix).unwrap(), &self.value_maps)
                    })
                    .collect()
            })
            .collect();

        self.engine
            .logp(&col_ixs, &data, &given, state_ixs_opt)
            .unwrap()
    }

    #[args(with_uncertainty = "true")]
    fn predict(&self, target: &PyAny, given: Option<&PyDict>, with_uncertainty: bool) -> Py<PyAny> {
        let col_ix = value_to_index(target, &self.col_indexer);
        let given = dict_to_given(given, &self.engine, &self.col_indexer, &self.value_maps);
        if with_uncertainty {
            let unc_type_opt = Some(PredictUncertaintyType::JsDivergence);
            let (pred, unc) = self.engine.predict(col_ix, &given, unc_type_opt).unwrap();
            let value = datum_to_value(pred, col_ix, &self.engine.codebook);
            Python::with_gil(|py| {
                let unc = unc.into_py(py);
                (value, unc).into_py(py)
            })
        } else {
            let (pred, _) = self.engine.predict(col_ix, &given, None).unwrap();
            datum_to_value(pred, col_ix, &self.engine.codebook)
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "pybraid")]
fn pybraid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}

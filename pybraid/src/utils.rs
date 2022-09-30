use std::collections::HashMap;

use braid::codebook::{Codebook, ColType};
use braid::{Datum, FType, Given, OracleT};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyTuple};

pub(crate) struct Indexer {
    pub to_ix: HashMap<String, usize>,
    pub to_name: HashMap<usize, String>,
}

impl Indexer {
    pub(crate) fn columns(codebook: &Codebook) -> Self {
        let mut to_ix: HashMap<String, usize> = HashMap::new();
        let mut to_name: HashMap<usize, String> = HashMap::new();
        codebook
            .col_metadata
            .iter()
            .enumerate()
            .for_each(|(ix, col_md)| {
                to_ix.insert(col_md.name.clone(), ix);
                to_name.insert(ix, col_md.name.clone());
            });

        Self { to_ix, to_name }
    }

    pub(crate) fn rows(codebook: &Codebook) -> Self {
        let mut to_ix: HashMap<String, usize> = HashMap::new();
        let mut to_name: HashMap<usize, String> = HashMap::new();
        codebook.row_names.iter().for_each(|(name, &ix)| {
            to_ix.insert(name.clone(), ix);
            to_name.insert(ix, name.clone());
        });

        Self { to_ix, to_name }
    }
}

pub(crate) fn list_to_pairs(pairs: &PyList, indexer: &Indexer) -> Vec<(usize, usize)> {
    pairs
        .iter()
        .map(|item| {
            item.downcast::<PyList>()
                .map(|ixs| {
                    assert_eq!(ixs.len(), 2);
                    let a = value_to_index(&ixs[0], indexer);
                    let b = value_to_index(&ixs[1], indexer);
                    (a, b)
                })
                .unwrap_or_else(|_| {
                    let ixs: &PyTuple = item.downcast().unwrap();
                    assert_eq!(ixs.len(), 2);
                    let a = value_to_index(&ixs[0], indexer);
                    let b = value_to_index(&ixs[1], indexer);
                    (a, b)
                })
        })
        .collect()
}

pub(crate) fn value_maps(codebook: &Codebook) -> HashMap<usize, HashMap<String, usize>> {
    codebook
        .col_metadata
        .iter()
        .enumerate()
        .filter_map(|(ix, col_md)| match col_md.coltype {
            ColType::Categorical {
                value_map: Some(ref value_map),
                ..
            } => {
                let revmap = value_map.iter().map(|(&k, v)| (v.clone(), k)).collect();
                Some((ix, revmap))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn datum_to_value(datum: Datum, ix: usize, codebook: &Codebook) -> Py<PyAny> {
    Python::with_gil(|py| match datum {
        Datum::Continuous(x) => x.to_object(py),
        Datum::Count(x) => x.to_object(py),
        Datum::Categorical(x) => {
            let coltype = &codebook.col_metadata[ix].coltype;
            match coltype {
                ColType::Categorical {
                    value_map: None, ..
                } => x.to_object(py),
                ColType::Categorical {
                    value_map: Some(ref value_map),
                    ..
                } => {
                    let s = value_map[&ix].as_str();
                    s.to_object(py)
                }
                _ => panic!("ColType for {ix} not compatible with Datum::Categorical"),
            }
        }
        _ => panic!("Unsupported Datum Type"),
    })
}

pub(crate) fn value_to_datum(
    val: &PyAny,
    ix: usize,
    ftype: FType,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> Datum {
    match ftype {
        FType::Continuous => {
            let f: &PyFloat = val.downcast().unwrap();
            Datum::Continuous(f.value())
        }
        FType::Categorical => {
            let x: u8 = val.downcast::<PyInt>().map_or_else(
                |_| {
                    let s: String = val.extract().unwrap();
                    let x = value_maps[&ix][&s];
                    x as u8
                },
                |i| {
                    let x: u8 = i.extract().unwrap();
                    x
                },
            );
            Datum::Categorical(x)
        }
        FType::Count => Datum::Count(val.extract().unwrap()),
        _ => panic!("Unsupported FType"),
    }
}

pub(crate) fn value_to_index(val: &PyAny, indexer: &Indexer) -> usize {
    val.extract::<usize>().unwrap_or_else(|_| {
        let s: &str = val.extract().unwrap();
        indexer.to_ix[s]
    })
}

pub(crate) fn column_indices(cols: &PyList, indexer: &Indexer) -> Vec<usize> {
    cols.iter()
        .map(|val| value_to_index(val, indexer))
        .collect()
}

pub(crate) fn dict_to_given(
    dict_opt: Option<&PyDict>,
    engine: &braid::Engine,
    indexer: &Indexer,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> Given {
    match dict_opt {
        None => Given::Nothing,
        Some(dict) if dict.is_empty() => Given::Nothing,
        Some(dict) => {
            let conditions = dict
                .iter()
                .map(|(key, value)| {
                    let ix = value_to_index(key, indexer);
                    let x = value_to_datum(value, ix, engine.ftype(ix).unwrap(), value_maps);
                    (ix, x)
                })
                .collect();

            Given::Conditions(conditions)
        }
    }
}

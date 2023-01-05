use std::collections::HashMap;

use braid::cc::alg::{ColAssignAlg, RowAssignAlg};
use braid::codebook::{Codebook, ColType};
use braid::{Datum, FType, Given, OracleT, StateTransition};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use regex::Regex;

fn parse_transition(trns: &str) -> StateTransition {
    let col_asgn_re =
        Regex::new(r"column_assignment\((gibbs|slice|finite_cpu)\)").unwrap();
    let row_asgn_re =
        Regex::new(r"row_assignment\((gibbs|slice|finite_cpu|sams)\)").unwrap();

    let col_match = col_asgn_re.captures(trns);
    let row_match = row_asgn_re.captures(trns);

    match (trns, col_match, row_match) {
        ("state_alpha", None, None) => StateTransition::StateAlpha,
        ("view_alphas", None, None) => StateTransition::ViewAlphas,
        ("component_params", None, None) => StateTransition::ComponentParams,
        ("feature_priors", None, None) => StateTransition::FeaturePriors,
        (_, Some(col_asgn), None) => {
            let alg = col_asgn.get(1).unwrap().as_str();
            match alg {
                "gibbs" => {
                    StateTransition::ColumnAssignment(ColAssignAlg::Gibbs)
                }
                "slice" => {
                    StateTransition::ColumnAssignment(ColAssignAlg::Slice)
                }
                "finite_cpu" => {
                    StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu)
                }
                _ => panic!("Invalid column assignment algorithm `{alg}`"),
            }
        }
        (_, None, Some(row_asgn)) => {
            let alg = row_asgn.get(1).unwrap().as_str();
            match alg {
                "sams" => StateTransition::RowAssignment(RowAssignAlg::Sams),
                "gibbs" => StateTransition::RowAssignment(RowAssignAlg::Gibbs),
                "slice" => StateTransition::RowAssignment(RowAssignAlg::Slice),
                "finite_cpu" => {
                    StateTransition::RowAssignment(RowAssignAlg::FiniteCpu)
                }
                _ => panic!("Invalid row assignment algorithm `{alg}`"),
            }
        }
        _ => panic!("Invalid transition `{trns}`"),
    }
}

pub(crate) fn parse_transitions(trns: &[String]) -> Vec<StateTransition> {
    trns.iter().map(|s| parse_transition(s)).collect()
}

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

pub(crate) fn list_to_pairs(
    pairs: &PyList,
    indexer: &Indexer,
) -> Vec<(usize, usize)> {
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

pub(crate) fn value_maps(
    codebook: &Codebook,
) -> HashMap<usize, HashMap<String, usize>> {
    codebook
        .col_metadata
        .iter()
        .enumerate()
        .filter_map(|(ix, col_md)| match col_md.coltype {
            ColType::Categorical {
                value_map: Some(ref value_map),
                ..
            } => {
                let revmap =
                    value_map.iter().map(|(&k, v)| (v.clone(), k)).collect();
                Some((ix, revmap))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn datum_to_value(
    datum: Datum,
    ix: usize,
    codebook: &Codebook,
) -> Py<PyAny> {
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
                    let s = value_map[&(x as usize)].as_str();
                    s.to_object(py)
                }
                _ => panic!(
                    "ColType for {ix} not compatible with Datum::Categorical"
                ),
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
        if let Some(ix) = indexer.to_ix.get(s) {
            *ix
        } else {
            panic!("Unknown value '{s}' for index");
        }
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
) -> Given<usize> {
    match dict_opt {
        None => Given::Nothing,
        Some(dict) if dict.is_empty() => Given::Nothing,
        Some(dict) => {
            let conditions = dict
                .iter()
                .map(|(key, value)| {
                    let ix = value_to_index(key, indexer);
                    let x = value_to_datum(
                        value,
                        ix,
                        engine.ftype(ix).unwrap(),
                        value_maps,
                    );
                    (ix, x)
                })
                .collect();

            Given::Conditions(conditions)
        }
    }
}

pub(crate) fn srs_to_strings(srs: &PyAny) -> Vec<String> {
    let list: &PyList = srs.call_method0("tolist").unwrap().extract().unwrap();

    list.iter()
        .map(|x| {
            let s: &PyString =
                x.call_method0("__repr__").unwrap().extract().unwrap();
            s.to_string()
        })
        .collect()
}

pub(crate) fn parts_to_insert_values(
    col_ixs: Vec<usize>,
    mut row_names: Vec<String>,
    mut values: Vec<Vec<Datum>>,
) -> Vec<braid::Row<String, usize>> {
    use braid::Value;
    row_names
        .drain(..)
        .zip(values.drain(..))
        .map(|(row_name, mut row)| {
            let values = col_ixs
                .iter()
                .zip(row.drain(..))
                .map(|(&col_ix, value)| Value { col_ix, value })
                .collect();

            braid::Row {
                row_ix: row_name,
                values,
            }
        })
        .collect()
}

fn values_to_data(
    data: &PyList,
    col_ixs: &[usize],
    engine: &braid::Engine,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> Vec<Vec<Datum>> {
    data.iter()
        .map(|row_any| {
            let row: &PyList = row_any.downcast().unwrap();
            col_ixs
                .iter()
                .zip(row.iter())
                .map(|(&ix, val)| {
                    value_to_datum(
                        val,
                        ix,
                        engine.ftype(ix).unwrap(),
                        value_maps,
                    )
                })
                .collect()
        })
        .collect()
}

fn df_to_values(
    df: &PyAny,
    indexer: &Indexer,
    engine: &braid::Engine,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> (Vec<usize>, Vec<String>, Vec<Vec<Datum>>) {
    Python::with_gil(|py| {
        let row_names = srs_to_strings(df.getattr("index").unwrap());
        let columns = df
            .getattr("columns")
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .to_object(py);
        let data = df
            .getattr("values")
            .unwrap()
            .call_method0("tolist")
            .unwrap();

        let data: &PyList = data.extract().unwrap();
        let columns: &PyList = columns.extract(py).unwrap();
        let col_ixs = column_indices(columns, indexer);
        let data = values_to_data(data, &col_ixs, engine, value_maps);
        (col_ixs, row_names, data)
    })
}

fn srs_to_column_values(
    srs: &PyAny,
    indexer: &Indexer,
    engine: &braid::Engine,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> (Vec<usize>, Vec<String>, Vec<Vec<Datum>>) {
    let data = srs.call_method0("to_frame").unwrap();

    df_to_values(data, indexer, engine, value_maps)
}

fn srs_to_row_values(
    srs: &PyAny,
    indexer: &Indexer,
    engine: &braid::Engine,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> (Vec<usize>, Vec<String>, Vec<Vec<Datum>>) {
    let data = srs
        .call_method0("to_frame")
        .unwrap()
        .call_method0("transpose")
        .unwrap();

    df_to_values(data, indexer, engine, value_maps)
}

pub(crate) fn pandas_to_logp_values(
    xs: &PyAny,
    indexer: &Indexer,
    engine: &braid::Engine,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> (Vec<usize>, Vec<String>, Vec<Vec<Datum>>) {
    let type_name = xs.get_type().name().unwrap();

    match type_name {
        "DataFrame" => df_to_values(xs, indexer, engine, value_maps),
        "Series" => srs_to_column_values(xs, indexer, engine, value_maps),
        _ => panic!("Unsupported value type"),
    }
}

pub(crate) fn pandas_to_insert_values(
    xs: &PyAny,
    indexer: &Indexer,
    engine: &braid::Engine,
    value_maps: &HashMap<usize, HashMap<String, usize>>,
) -> (Vec<usize>, Vec<String>, Vec<Vec<Datum>>) {
    let type_name = xs.get_type().name().unwrap();

    match type_name {
        "DataFrame" => df_to_values(xs, indexer, engine, value_maps),
        "Series" => srs_to_row_values(xs, indexer, engine, value_maps),
        _ => panic!("Unsupported value type"),
    }
}

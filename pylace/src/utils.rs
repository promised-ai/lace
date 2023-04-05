use std::collections::HashMap;

use lace::codebook::{Codebook, ColType, ValueMap};
use lace::{ColumnIndex, Datum, FType, Given, OracleT};
use polars::frame::DataFrame;
use polars::prelude::NamedFrom;
use polars::series::Series;
use pyo3::exceptions::{
    PyIndexError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple,
};

use crate::df::{PyDataFrame, PySeries};

pub(crate) fn to_pyerr(err: impl std::error::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err}"))
}

const NONE: Option<f64> = None;

pub(crate) struct MiArgs {
    pub(crate) n_mc_samples: usize,
    pub(crate) mi_type: String,
}

pub(crate) struct RowsimArgs<'a> {
    pub(crate) wrt: Option<&'a PyAny>,
    pub(crate) col_weighted: bool,
}

impl Default for MiArgs {
    fn default() -> Self {
        Self {
            n_mc_samples: 1_000,
            mi_type: String::from("iqr"),
        }
    }
}

impl<'a> Default for RowsimArgs<'a> {
    fn default() -> Self {
        Self {
            wrt: None,
            col_weighted: false,
        }
    }
}

pub(crate) fn mi_args_from_dict(dict: &PyDict) -> PyResult<MiArgs> {
    let n_mc_samples: Option<usize> = dict
        .get_item("n_mc_samples")
        .map(|any| any.extract::<usize>())
        .transpose()?;

    let mi_type: Option<String> = dict
        .get_item("mi_type")
        .map(|any| any.extract::<String>())
        .transpose()?;

    Ok(MiArgs {
        n_mc_samples: n_mc_samples.unwrap_or(1_000),
        mi_type: mi_type.unwrap_or_else(|| String::from("iqr")),
    })
}

pub(crate) fn rowsim_args_from_dict<'a>(
    dict: &'a PyDict,
) -> PyResult<RowsimArgs<'a>> {
    let col_weighted: Option<bool> = dict
        .get_item("col_weighted")
        .map(|any| any.extract::<bool>())
        .transpose()?;

    let wrt: Option<&PyAny> = dict.get_item("wrt");

    Ok(RowsimArgs {
        wrt,
        col_weighted: col_weighted.unwrap_or(false),
    })
}

macro_rules! srs_from_vec {
    ($values: ident, $name: expr, $xtype: ty, $variant: ident) => {{
        let xs: Vec<Option<$xtype>> = $values.drain(..).map(|x| {
            match x {
                // as is used to convert the u8 in categorical to u32 because
                // polars NamedFrom doesn't appear to support u8 or u16 Vecs
                Datum::$variant(x) => Some(x as $xtype),
                _ => None,
            }
        }).collect();
        Series::new($name, xs)
    }};
}

macro_rules! cat_srs_from_vec {
    ($values: ident, $name: expr, $xtype: ty, $variant: ident) => {{
        let xs: Vec<Option<$xtype>> = $values
            .drain(..)
            .map(|x| match x {
                Datum::Categorical(lace::Category::$variant(x)) => Some(x),
                _ => None,
            })
            .collect();
        Series::new($name, xs)
    }};
}

pub(crate) fn vec_to_srs(
    mut values: Vec<Datum>,
    col_ix: usize,
    ftype: FType,
    codebook: &Codebook,
) -> PyResult<PySeries> {
    let col_md = &codebook.col_metadata[col_ix];
    let name = col_md.name.as_str();

    match ftype {
        FType::Binary => Ok(srs_from_vec!(values, name, bool, Binary)),
        FType::Continuous => Ok(srs_from_vec!(values, name, f64, Continuous)),
        FType::Categorical => {
            let repr = CategoricalRepr::from_codebook(col_ix, codebook);
            match repr {
                CategoricalRepr::Int => {
                    Ok(cat_srs_from_vec!(values, name, u8, U8))
                }
                CategoricalRepr::String => {
                    Ok(cat_srs_from_vec!(values, name, String, String))
                }
                CategoricalRepr::Bool => {
                    Ok(cat_srs_from_vec!(values, name, bool, Bool))
                }
            }
        }
        FType::Count => Ok(srs_from_vec!(values, name, u32, Count)),
        // ftype => Err(PyErr::new::<PyValueError, _>(format!(
        //     "Simulated unsupported ftype: {ftype:?}"
        // ))),
    }
    .map(PySeries)
}

macro_rules! srs_from_simulate {
    ($values: ident, $i: ident, $name: expr, $xtype: ty, $variant: ident) => {{
        let xs: Vec<Option<$xtype>> = $values.iter().map(|row| {
            match row[$i] {
                // as is used to convert the u8 in categorical to u32 because
                // polars NamedFrom doesn't appear to support u8 or u16 Vecs
                Datum::$variant(x) => Some(x as $xtype),
                _ => None,
            }
        }).collect();
        Series::new($name, xs)
    }};
}

macro_rules! cat_srs_from_simulate {
    ($values: ident, $i: ident, $name: expr, $xtype: ty, $variant: ident) => {{
        let xs: Vec<Option<$xtype>> = $values
            .iter()
            .map(|row| match row[$i] {
                Datum::Categorical(lace::Category::$variant(ref x)) => {
                    Some(x.clone())
                }
                _ => None,
            })
            .collect();
        Series::new($name, xs)
    }};
}

pub(crate) fn simulate_to_df(
    values: Vec<Vec<Datum>>,
    ftypes: &[FType],
    col_ixs: &[usize],
    indexer: &Indexer,
    codebook: &Codebook,
) -> PyResult<PyDataFrame> {
    let mut df = DataFrame::default();

    for (i, col_ix) in col_ixs.iter().enumerate() {
        let name = indexer.to_name[col_ix].as_str();
        let srs: Series = match ftypes[*col_ix] {
            FType::Binary => Ok::<Series, PyErr>(srs_from_simulate!(
                values, i, name, bool, Binary
            )),
            FType::Continuous => {
                Ok(srs_from_simulate!(values, i, name, f64, Continuous))
            }
            FType::Categorical => {
                let repr = CategoricalRepr::from_codebook(*col_ix, codebook);
                match repr {
                    CategoricalRepr::Int => {
                        Ok(cat_srs_from_simulate!(values, i, name, u8, U8))
                    }
                    CategoricalRepr::String => Ok(cat_srs_from_simulate!(
                        values, i, name, String, String
                    )),
                    CategoricalRepr::Bool => {
                        Ok(cat_srs_from_simulate!(values, i, name, bool, Bool))
                    }
                }
            }
            FType::Count => Ok(srs_from_simulate!(values, i, name, u32, Count)),
            // ftype => Err(PyErr::new::<PyValueError, _>(format!(
            //     "Simulated unsupported ftype: {ftype:?}"
            // ))),
        }?;
        df.with_column(srs).map_err(|err| {
            PyErr::new::<PyRuntimeError, _>(format!(
                "Failed to append column tp df: {err}"
            ))
        })?;
    }
    Ok(PyDataFrame(df))
}

pub(crate) fn str_to_mitype(mi_type: &str) -> PyResult<lace::MiType> {
    match mi_type.to_lowercase().as_str() {
        "unnormed" => Ok(lace::MiType::UnNormed),
        "normed" => Ok(lace::MiType::Normed),
        "iqr" => Ok(lace::MiType::Iqr),
        "voi" => Ok(lace::MiType::Voi),
        "jaccard" => Ok(lace::MiType::Jaccard),
        "linfoot" => Ok(lace::MiType::Linfoot),
        "pearson" => Ok(lace::MiType::Pearson),
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "Invalid mi_type: {}",
            mi_type
        ))),
    }
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

pub(crate) fn pairs_list_iter<'a>(
    pairs: &'a PyList,
    indexer: &'a Indexer,
) -> impl Iterator<Item = PyResult<(usize, usize)>> + 'a {
    pairs.iter().map(|item| {
        item.downcast::<PyList>()
            .map(|ixs| {
                if ixs.len() != 2 {
                    Err(PyErr::new::<PyValueError, _>(
                        "A pair consists of two items",
                    ))
                } else {
                    value_to_index(&ixs[0], indexer).and_then(|a| {
                        value_to_index(&ixs[1], indexer).map(|b| (a, b))
                    })
                }
            })
            .unwrap_or_else(|_| {
                let ixs: &PyTuple = item.downcast().unwrap();
                if ixs.len() != 2 {
                    Err(PyErr::new::<PyValueError, _>(
                        "A pair consists of two items",
                    ))
                } else {
                    value_to_index(&ixs[0], indexer).and_then(|a| {
                        value_to_index(&ixs[1], indexer).map(|b| (a, b))
                    })
                }
            })
    })
}

pub(crate) fn list_to_pairs<'a>(
    pairs: &'a PyList,
    indexer: &'a Indexer,
) -> PyResult<Vec<(usize, usize)>> {
    pairs_list_iter(pairs, indexer).collect()
}

enum CategoricalRepr {
    String,
    Int,
    Bool,
}

impl CategoricalRepr {
    fn from_codebook(col_ix: usize, codebook: &Codebook) -> CategoricalRepr {
        if let Some(value_map) = codebook.value_map(col_ix) {
            match value_map {
                ValueMap::String(_) => CategoricalRepr::String,
                ValueMap::U8(_) => CategoricalRepr::Int,
                ValueMap::Bool => CategoricalRepr::Bool,
            }
        } else {
            panic!("ColType for {col_ix} is not Categorical")
        }
    }
}

pub(crate) fn datum_to_value(datum: Datum) -> PyResult<Py<PyAny>> {
    use lace::Category;
    Python::with_gil(|py| match datum {
        Datum::Continuous(x) => Ok(x.to_object(py)),
        Datum::Count(x) => Ok(x.to_object(py)),
        Datum::Categorical(Category::U8(x)) => Ok(x.to_object(py)),
        Datum::Categorical(Category::Bool(x)) => Ok(x.to_object(py)),
        Datum::Categorical(Category::String(x)) => Ok(x.to_object(py)),
        Datum::Missing => Ok(NONE.to_object(py)),
        x => Err(PyErr::new::<PyValueError, _>(format!(
            "Unsupported datum: {:?}",
            x
        ))),
    })
}

fn pyany_to_category(val: &PyAny) -> PyResult<lace::Category> {
    use lace::Category;
    let name = val.get_type().name()?;

    match name {
        "int" => {
            let x = val.downcast::<PyInt>()?.extract::<u8>()?;
            Ok(Category::U8(x))
        }
        "bool" => {
            let x = val.downcast::<PyBool>()?.extract::<bool>()?;
            Ok(Category::Bool(x))
        }
        "str" => {
            let x = val.downcast::<PyString>()?.extract::<String>()?;
            Ok(Category::String(x))
        }
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "Cannot convert {name} into Category"
        ))),
    }
}

pub(crate) fn value_to_datum(val: &PyAny, ftype: FType) -> PyResult<Datum> {
    if val.is_none() {
        return Ok(Datum::Missing);
    }

    match ftype {
        FType::Continuous => {
            let x: f64 = {
                let f: &PyFloat = val.downcast().unwrap();
                f.value()
            };
            if x.is_nan() {
                Ok(Datum::Missing)
            } else {
                Ok(Datum::Continuous(x))
            }
        }
        FType::Categorical => {
            let x = pyany_to_category(val)?;
            Ok(Datum::Categorical(x))
        }
        FType::Count => Ok(Datum::Count(val.extract().unwrap())),
        ftype => Err(PyErr::new::<PyValueError, _>(format!(
            "Unsupported ftype: {:?}",
            ftype
        ))),
    }
}

pub(crate) fn value_to_name(
    val: &PyAny,
    indexer: &Indexer,
) -> PyResult<String> {
    val.extract::<String>().or_else(|_| {
        let ix: usize = val.extract().unwrap();
        if let Some(name) = indexer.to_name.get(&ix) {
            Ok(name.to_owned())
        } else {
            Err(PyErr::new::<PyIndexError, _>(format!("No index {}", ix)))
        }
    })
}

pub(crate) fn value_to_index(
    val: &PyAny,
    indexer: &Indexer,
) -> PyResult<usize> {
    val.extract::<usize>().or_else(|_| {
        let s: &str = val.extract().unwrap();
        if let Some(ix) = indexer.to_ix.get(s) {
            Ok(*ix)
        } else {
            Err(PyErr::new::<PyIndexError, _>(format!(
                "Unknown value '{s}' for index"
            )))
        }
    })
}

pub(crate) fn pyany_to_indices(
    cols: &PyAny,
    indexer: &Indexer,
) -> PyResult<Vec<usize>> {
    cols.iter()?
        .map(|res| res.and_then(|val| value_to_index(val, indexer)))
        .collect()
}

pub(crate) fn dict_to_given(
    dict_opt: Option<&PyDict>,
    engine: &lace::Engine,
    indexer: &Indexer,
) -> PyResult<Given<usize>> {
    match dict_opt {
        None => Ok(Given::Nothing),
        Some(dict) if dict.is_empty() => Ok(Given::Nothing),
        Some(dict) => {
            let conditions = dict
                .iter()
                .map(|(key, value)| {
                    value_to_index(key, indexer).and_then(|ix| {
                        value_to_datum(value, engine.ftype(ix).unwrap())
                            .map(|x| (ix, x))
                    })
                })
                .collect::<PyResult<Vec<(usize, Datum)>>>()?;

            Ok(Given::Conditions(conditions))
        }
    }
}

pub(crate) fn srs_to_strings(srs: &PyAny) -> PyResult<Vec<String>> {
    let list: &PyList = srs.call_method0("to_list").unwrap().extract().unwrap();

    list.iter()
        .map(|x| {
            x.extract::<String>()
            // x.call_method0("__repr__").unwrap().extract().unwrap();
            // s.to_string()
        })
        .collect::<PyResult<Vec<String>>>()
}

pub(crate) fn parts_to_insert_values(
    col_ixs: Vec<usize>,
    mut row_names: Vec<String>,
    mut values: Vec<Vec<Datum>>,
) -> Vec<lace::Row<String, usize>> {
    use lace::Value;
    row_names
        .drain(..)
        .zip(values.drain(..))
        .map(|(row_name, mut row)| {
            let values = col_ixs
                .iter()
                .zip(row.drain(..))
                .map(|(&col_ix, value)| Value { col_ix, value })
                .collect();

            lace::Row {
                row_ix: row_name,
                values,
            }
        })
        .collect()
}

pub(crate) fn pyany_to_data(
    data: &PyAny,
    ftype: FType,
) -> PyResult<Vec<Datum>> {
    data.iter()?
        .map(|res| res.and_then(|val| value_to_datum(val, ftype)))
        .collect()
}

fn process_row_dict(
    row_dict: &PyDict,
    col_ixs: &[usize],
    engine: &lace::Engine,
) -> Result<Vec<Datum>, PyErr> {
    let index_to_values_map: HashMap<_, _> = row_dict
        .iter()
        .map(|(name_any, value_any)| {
            let column_name: &PyString = name_any.downcast().unwrap();
            let column_name = column_name.to_str().unwrap();
            let column_index = column_name.col_ix(&engine.codebook).unwrap();
            (column_index, value_any)
        })
        .collect();

    // Process the `dict`s values by the indexes in the col_ixs variable
    // This assures that only columns specified in col_ixs is retrieved,
    // and that no column is missing in the data
    let all_column_datums = col_ixs
        .iter()
        .map(|&column_index| {
            let &value_any = index_to_values_map.get(&column_index).unwrap();
            value_to_datum(value_any, engine.ftype(column_index).unwrap())
        })
        .collect();

    all_column_datums
}

// Works on list of dicts
fn values_to_data(
    data: &PyList,
    col_ixs: &[usize],
    engine: &lace::Engine,
) -> PyResult<Vec<Vec<Datum>>> {
    data.iter()
        .map(|row_any| {
            let row_dict: &PyDict = row_any.downcast().unwrap();

            process_row_dict(row_dict, col_ixs, engine)
        })
        .collect()
}

pub(crate) struct DataFrameComponents {
    pub col_ixs: Vec<usize>,
    pub row_names: Option<Vec<String>>,
    pub values: Vec<Vec<Datum>>,
}

// FIXME: pass the 'py' in so that we can handle errors bettter. The
// `Python::with_gil` thing makes using `?` a pain.
fn df_to_values(
    df: &PyAny,
    indexer: &Indexer,
    engine: &lace::Engine,
) -> PyResult<DataFrameComponents> {
    let row_names = if df.hasattr("index")? {
        let index = df.getattr("index")?;
        srs_to_strings(index).ok()
    } else {
        df.call_method1("__getitem__", ("index",))
            .and_then(srs_to_strings)
            .ok()
    };

    Python::with_gil(|py| {
        let (columns, data) = {
            let columns = df.getattr("columns").unwrap();
            if columns.get_type().name().unwrap().contains("Index") {
                // Is a Pandas dataframe
                let cols =
                    columns.call_method0("tolist").unwrap().to_object(py);
                let kwargs = PyDict::new(py);
                kwargs.set_item("orient", "records").unwrap();
                let data = df.call_method("to_dict", (), Some(kwargs)).unwrap();
                (cols, data)
            } else {
                // Is a Polars dataframe
                let list = columns.downcast::<PyList>().unwrap();
                let has_index = list
                    .iter()
                    .any(|s| s.extract::<&str>().unwrap() == "index");

                let df = if has_index {
                    // remove the index column label
                    list.call_method1("remove", ("index",)).unwrap();
                    // remove the index column from the data
                    df.call_method1("drop", ("index",)).unwrap()
                } else {
                    df
                };

                let data = df.call_method0("to_dicts").unwrap();
                (list.to_object(py), data)
            }
        };

        let data: &PyList = data.extract().unwrap();
        let columns: &PyList = columns.extract(py).unwrap();
        pyany_to_indices(columns, indexer)
            .and_then(|col_ixs| {
                values_to_data(data, &col_ixs, engine)
                    .map(|data| (col_ixs, data))
            })
            .map(|(col_ixs, values)| DataFrameComponents {
                col_ixs,
                row_names,
                values,
            })
    })
}

fn srs_to_column_values(
    srs: &PyAny,
    indexer: &Indexer,
    engine: &lace::Engine,
) -> PyResult<DataFrameComponents> {
    let data = srs.call_method0("to_frame").unwrap();

    df_to_values(data, indexer, engine)
}

fn srs_to_row_values(
    srs: &PyAny,
    indexer: &Indexer,
    engine: &lace::Engine,
) -> PyResult<DataFrameComponents> {
    let data = srs
        .call_method0("to_frame")
        .unwrap()
        .call_method0("transpose")
        .unwrap();

    df_to_values(data, indexer, engine)
}

pub(crate) fn pandas_to_logp_values(
    xs: &PyAny,
    indexer: &Indexer,
    engine: &lace::Engine,
) -> PyResult<DataFrameComponents> {
    let type_name = xs.get_type().name().unwrap();

    match type_name {
        "DataFrame" => df_to_values(xs, indexer, engine),
        "Series" => srs_to_column_values(xs, indexer, engine),
        t => Err(PyErr::new::<PyTypeError, _>(format!(
            "Unsupported type: {t}"
        ))),
    }
}

pub(crate) fn pandas_to_insert_values(
    xs: &PyAny,
    indexer: &Indexer,
    engine: &lace::Engine,
) -> PyResult<DataFrameComponents> {
    let type_name = xs.get_type().name().unwrap();

    match type_name {
        "DataFrame" => df_to_values(xs, indexer, engine),
        "Series" => srs_to_row_values(xs, indexer, engine),
        t => Err(PyErr::new::<PyTypeError, _>(format!(
            "Unsupported type: {t}"
        ))),
    }
}

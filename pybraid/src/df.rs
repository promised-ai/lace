use arrow2::ffi;
use polars::frame::DataFrame;
use polars::prelude::{ArrayRef, ArrowField};
use polars::series::Series;
use pyo3::ffi::Py_uintptr_t;
use pyo3::types::PyModule;
use pyo3::{IntoPy, PyObject, PyResult, Python, ToPyObject};

pub enum DataFrameLike {
    DataFrame(DataFrame),
    Series(Series),
    Float(f64),
    UInt(u32),
    Int(i64),
    String(String),
}

impl IntoPy<PyObject> for DataFrameLike {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Self::DataFrame(inner) => PyDataFrame(inner).into_py(py),
            Self::Series(inner) => PySeries(inner).into_py(py),
            Self::Float(inner) => inner.into_py(py),
            Self::UInt(inner) => inner.into_py(py),
            Self::Int(inner) => inner.into_py(py),
            Self::String(inner) => inner.into_py(py),
        }
    }
}

#[repr(transparent)]
pub struct PySeries(pub Series);

#[repr(transparent)]
pub struct PyDataFrame(pub DataFrame);

/// Arrow array to Python.
pub(crate) fn to_py_array(
    array: ArrayRef,
    py: Python,
    pyarrow: &PyModule,
) -> PyResult<PyObject> {
    let schema = Box::new(ffi::export_field_to_c(&ArrowField::new(
        "",
        array.data_type().clone(),
        true,
    )));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    Ok(array.to_object(py))
}

impl IntoPy<PyObject> for PySeries {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = self.0.rechunk();
        let name = s.name();
        let arr = s.to_arrow(0);
        let pyarrow = py.import("pyarrow").expect("pyarrow not installed");
        let polars = py.import("polars").expect("polars not installed");

        let arg = to_py_array(arr, py, pyarrow).unwrap();
        let s = polars.call_method1("from_arrow", (arg,)).unwrap();
        let s = s.call_method1("rename", (name,)).unwrap();
        s.to_object(py)
    }
}

impl IntoPy<PyObject> for PyDataFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let pyseries = self
            .0
            .get_columns()
            .iter()
            .map(|s| PySeries(s.clone()).into_py(py))
            .collect::<Vec<_>>();

        let polars = py.import("polars").expect("polars not installed");
        let df_object = polars.call_method1("DataFrame", (pyseries,)).unwrap();
        df_object.into_py(py)
    }
}

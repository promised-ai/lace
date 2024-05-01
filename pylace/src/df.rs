use polars::frame::DataFrame;
use polars::prelude::{ArrayRef, ArrowField, PolarsError};
use polars::series::Series;
use polars_arrow::ffi;
use pyo3::exceptions::{PyException, PyIOError, PyValueError};
use pyo3::ffi::Py_uintptr_t;
use pyo3::types::{PyAnyMethods, PyModule};
use pyo3::{
    create_exception, Bound, FromPyObject, IntoPy, PyAny, PyErr, PyObject,
    PyResult, Python, ToPyObject,
};

#[derive(Debug)]
pub struct DataFrameError(PolarsError);

impl From<PolarsError> for DataFrameError {
    fn from(value: PolarsError) -> Self {
        Self(value)
    }
}

impl std::convert::From<DataFrameError> for PyErr {
    fn from(err: DataFrameError) -> PyErr {
        match &err.0 {
            PolarsError::ComputeError(err) => {
                ComputeError::new_err(err.to_string())
            }
            PolarsError::NoData(err) => NoDataError::new_err(err.to_string()),
            PolarsError::ShapeMismatch(err) => {
                ShapeError::new_err(err.to_string())
            }
            PolarsError::SchemaMismatch(err) => {
                SchemaError::new_err(err.to_string())
            }
            PolarsError::Io(err) => PyIOError::new_err(err.to_string()),
            PolarsError::InvalidOperation(err) => {
                PyValueError::new_err(err.to_string())
            }
            PolarsError::Duplicate(err) => {
                DuplicateError::new_err(err.to_string())
            }
            PolarsError::ColumnNotFound(err) => {
                ColumnNotFound::new_err(err.to_string())
            }
            PolarsError::SchemaFieldNotFound(err) => {
                SchemaFieldNotFound::new_err(err.to_string())
            }
            PolarsError::StructFieldNotFound(err) => {
                StructFieldNotFound::new_err(err.to_string())
            }
            PolarsError::StringCacheMismatch(err) => {
                StringCacheMismatch::new_err(err.to_string())
            }
            PolarsError::OutOfBounds(_) => {
                OutOfBounds::new_err(err.0.to_string())
            }
        }
    }
}

#[allow(dead_code)]
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
#[derive(Clone)]
pub struct PySeries(pub Series);

fn array_to_rust(obj: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular,
    // `_export_to_c` can go out of bounds
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref())
            .map_err(DataFrameError::from)?;
        let array = ffi::import_array_from_c(*array, field.data_type)
            .map_err(DataFrameError::from)?;
        Ok(array)
    }
}

impl<'a> FromPyObject<'a> for PySeries {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let ob = ob.call_method0("rechunk")?;

        let name = ob.getattr("name")?;
        let name = name.str()?.to_str()?;

        let arr = ob.call_method0("to_arrow")?;
        let arr = array_to_rust(arr)?;
        Ok(PySeries(
            Series::try_from((name, arr)).map_err(DataFrameError::from)?,
        ))
    }
}

#[repr(transparent)]
pub struct PyDataFrame(pub DataFrame);

impl<'a> FromPyObject<'a> for PyDataFrame {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let series = ob.call_method0("get_columns")?;
        let n = ob.getattr("width")?.extract::<usize>()?;
        let mut columns = Vec::with_capacity(n);
        for pyseries in series.iter()? {
            let pyseries = pyseries?;
            let s = pyseries.extract::<PySeries>()?.0;
            columns.push(s);
        }
        Ok(PyDataFrame(DataFrame::new_no_checks(columns)))
    }
}

/// Arrow array to Python.
pub(crate) fn to_py_array(
    array: ArrayRef,
    py: Python,
    pyarrow: &Bound<PyModule>,
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

// TODO: When https://github.com/PyO3/pyo3/issues/1813 is solved, implement a
// failable version.
impl IntoPy<PyObject> for PySeries {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = self.0.rechunk();
        let name = s.name();
        let arr = s.to_arrow(0);
        let pyarrow =
            py.import_bound("pyarrow").expect("pyarrow not installed");
        let polars = py.import_bound("polars").expect("polars not installed");

        let arg = to_py_array(arr, py, &pyarrow).unwrap();
        let s = polars.call_method1("from_arrow", (arg,)).unwrap();
        let s = s.call_method1("rename", (name,)).unwrap();
        s.to_object(py)
    }
}

// TODO: When https://github.com/PyO3/pyo3/issues/1813 is solved, implement a
// failable version.
impl IntoPy<PyObject> for PyDataFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let pyseries = self
            .0
            .get_columns()
            .iter()
            .map(|s| PySeries(s.clone()).into_py(py))
            .collect::<Vec<_>>();

        let polars = py.import_bound("polars").expect("polars not installed");
        let df_object = polars.call_method1("DataFrame", (pyseries,)).unwrap();
        df_object.into_py(py)
    }
}

create_exception!(exceptions, ColumnNotFound, PyException);
create_exception!(exceptions, SchemaFieldNotFound, PyException);
create_exception!(exceptions, StructFieldNotFound, PyException);
create_exception!(exceptions, ComputeError, PyException);
create_exception!(exceptions, NoDataError, PyException);
create_exception!(exceptions, ArrowErrorException, PyException);
create_exception!(exceptions, ShapeError, PyException);
create_exception!(exceptions, SchemaError, PyException);
create_exception!(exceptions, DuplicateError, PyException);
create_exception!(exceptions, StringCacheMismatch, PyException);
create_exception!(exceptions, OutOfBounds, PyException);

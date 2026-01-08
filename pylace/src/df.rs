use polars::frame::DataFrame;
use polars::prelude::ArrayRef;
use polars::prelude::CompatLevel;
use polars::prelude::PolarsError;
use polars::series::Series;
use polars_arrow::ffi;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyIOError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::types::PyAnyMethods;
use pyo3::types::PyList;
use pyo3::types::PyModule;
use pyo3::types::PyStringMethods;
use pyo3::Bound;
use pyo3::FromPyObject;
use pyo3::IntoPyObject;
use pyo3::PyAny;
use pyo3::PyErr;
use pyo3::PyResult;
use pyo3::Python;

#[derive(Debug)]
pub struct DataFrameError(PolarsError);

impl From<PolarsError> for DataFrameError {
    fn from(value: PolarsError) -> Self {
        Self(value)
    }
}

impl std::convert::From<DataFrameError> for PyErr {
    fn from(err: DataFrameError) -> Self {
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
            PolarsError::IO { error, .. } => {
                PyIOError::new_err(error.to_string())
            }
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
            _ => {
                todo!()
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

/*
impl IntoPy<Py<PyAny>> for DataFrameLike {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
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
*/

impl<'py> IntoPyObject<'py> for DataFrameLike {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(
        self,
        py: Python<'py>,
    ) -> Result<Self::Output, Self::Error> {
        match self {
            Self::DataFrame(inner) => PyDataFrame(inner).into_pyobject(py),
            Self::Series(inner) => PySeries(inner).into_pyobject(py),
            Self::Float(inner) => {
                Ok(inner.into_pyobject(py).expect("Infalliable").into_any())
            }
            Self::UInt(inner) => {
                Ok(inner.into_pyobject(py).expect("Infalliable").into_any())
            }
            Self::Int(inner) => {
                Ok(inner.into_pyobject(py).expect("Infalliable").into_any())
            }
            Self::String(inner) => {
                Ok(inner.into_pyobject(py).expect("Infalliable").into_any())
            }
        }
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct PySeries(pub Series);

fn array_to_rust(obj: &Bound<PyAny>) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &raw const *array;
    let schema_ptr = &raw const *schema;

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
        let array = ffi::import_array_from_c(*array, field.dtype)
            .map_err(DataFrameError::from)?;
        Ok(array)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySeries {
    fn extract(
        obj: pyo3::Borrowed<'a, 'py, PyAny>,
    ) -> Result<Self, Self::Error> {
        use polars::prelude::PlSmallStr;
        let obj = obj.call_method0("rechunk")?;

        let name = obj.getattr("name")?;
        let name = PlSmallStr::from(name.str()?.to_str()?);

        let arr = obj.call_method0("to_arrow")?;
        let arr = array_to_rust(&arr)?;
        Ok(Self(
            Series::try_from((name, arr)).map_err(DataFrameError::from)?,
        ))
    }

    type Error = PyErr;
}

#[repr(transparent)]
pub struct PyDataFrame(pub DataFrame);

impl<'a, 'py> FromPyObject<'a, 'py> for PyDataFrame {
    type Error = PyErr;

    fn extract(
        obj: pyo3::Borrowed<'a, 'py, PyAny>,
    ) -> Result<Self, Self::Error> {
        let series: Bound<'_, PyList> =
            obj.call_method0("get_columns")?.cast_into()?;
        let n: usize = obj.getattr("width")?.extract::<usize>()?;
        let mut columns = Vec::with_capacity(n);
        for pyseries in series.try_iter()? {
            let pyseries = pyseries?;
            let s = pyseries.extract::<PySeries>()?.0;
            columns.push(s.into());
        }
        Ok(Self(DataFrame::new(columns).unwrap()))
    }
}

/// Arrow array to Python.
fn to_py_array<'py>(
    array: Box<dyn polars_arrow::array::Array>,
    py: Python<'py>,
    pyarrow: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyAny>> {
    use polars_arrow::datatypes::Field;
    use polars_arrow::ffi::export_array_to_c;
    use polars_arrow::ffi::export_field_to_c;

    // Build schema
    let dtype = array.dtype().clone();
    let field = Field::new("".into(), dtype, true);

    // Export to C Arrow
    let schema = Box::new(export_field_to_c(&field));
    let array = Box::new(export_array_to_c(array));
    let schema_ptr: *const polars_arrow::ffi::ArrowSchema = &raw const *schema;
    let array_ptr: *const polars_arrow::ffi::ArrowArray = &raw const *array;

    let py_array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as usize, schema_ptr as usize),
    )?;

    py_array.into_pyobject(py).map_err(Into::into)
}

impl<'py> IntoPyObject<'py> for PySeries {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(
        self,
        py: Python<'py>,
    ) -> Result<Self::Output, Self::Error> {
        let s = self.0.rechunk();
        let name = s.name().as_str();
        let array = s.to_arrow(0, CompatLevel::newest());
        let pyarrow = py.import("pyarrow")?;
        let polars = py.import("polars")?;

        let argument = to_py_array(array, py, &pyarrow)?;
        let s = polars.call_method1("from_arrow", (argument,))?;
        let s = s.call_method1("rename", (name,))?;
        Ok(s)
    }
}

impl<'py> IntoPyObject<'py> for PyDataFrame {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(
        self,
        py: Python<'py>,
    ) -> Result<Self::Output, Self::Error> {
        let pyseries = self
            .0
            .get_columns()
            .iter()
            .map(|s| {
                PySeries(s.clone().as_materialized_series_maintain_scalar())
                    .into_pyobject(py)
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        let polars = py.import("polars")?;
        let df_object = polars.call_method1("DataFrame", (pyseries,))?;
        Ok(df_object)
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

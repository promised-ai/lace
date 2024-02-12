/// Update Handler and associated tooling for `CoreEngine.update` in Python.
use std::sync::{Arc, Mutex};

use lace::cc::state::State;
use lace::update_handler::UpdateHandler;
use lace::EngineUpdateConfig;
use pyo3::{pyclass, IntoPy, Py, PyAny};

/// Python version of `EngineUpdateConfig`.
#[derive(Clone, Debug)]
#[pyclass(frozen, get_all)]
pub struct PyEngineUpdateConfig {
    /// Maximum number of iterations to run.
    pub n_iters: usize,
    /// Number of iterations after which each state should be saved
    pub checkpoint: Option<usize>,
    /// Number of states
    pub n_states: usize,
}

/// An `UpdateHandler` which wraps a Python Object.
#[derive(Debug, Clone)]
pub struct PyUpdateHandler {
    handler: Arc<Mutex<Py<PyAny>>>,
}

impl PyUpdateHandler {
    /// Create a new `PyUpdateHandler` from a Python Object
    pub fn new(handler: Py<PyAny>) -> Self {
        Self {
            handler: Arc::new(Mutex::new(handler)),
        }
    }
}

macro_rules! pydict {
    ($py: expr, $($key:tt : $val:expr),* $(,)?) => {{

        let map = pyo3::types::PyDict::new($py);
        $(
            let _ = map.set_item($key, $val.into_py($py))
                .expect("Should be able to set item in PyDict");
        )*
        map
    }};
}

macro_rules! call_pyhandler_noret {
    ($self: ident, $func_name: tt, $($key: tt : $val: expr),* $(,)?) => {{
        let handler = $self
            .handler
            .lock()
            .expect("Should be able to get a lock for the PyUpdateHandler");

        ::pyo3::Python::with_gil(|py| {
            let kwargs = pydict!(
                py,
                $($key: $val),*
            );

            handler
                .call_method(py, $func_name, (), kwargs.into())
                .expect("Expected python call_method to return successfully");
        })
    }};
}

macro_rules! call_pyhandler_ret {
    ($self: ident, $func_name: tt, $($key: tt : $val: expr),* $(,)?) => {{
        let handler = $self
            .handler
            .lock()
            .expect("Should be able to get a lock for the PyUpdateHandler");

        ::pyo3::Python::with_gil(|py| {
            let kwargs = pydict!(
                py,
                $($key: $val),*
            );

            handler
                .call_method(py, $func_name, (), kwargs.into())
                .expect("Expected python call_method to return successfully")
                .extract(py)
                .expect("Failed to extract expected type")
        })
    }};
}

impl UpdateHandler for PyUpdateHandler {
    fn global_init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
        call_pyhandler_noret!(
            self,
            "global_init",
            "config": PyEngineUpdateConfig {
                n_iters: config.n_iters,
                checkpoint: config.checkpoint,
                n_states: states.len(),
            }
        );
    }

    fn new_state_init(&mut self, state_id: usize, _state: &State) {
        call_pyhandler_noret!(
            self,
            "new_state_init",
            "state_id": state_id,
        );
    }

    fn state_updated(&mut self, state_id: usize, _state: &State) {
        call_pyhandler_noret!(
            self,
            "state_updated",
            "state_id": state_id,
        );
    }

    fn state_complete(&mut self, state_id: usize, _state: &State) {
        call_pyhandler_noret!(
            self,
            "state_complete",
            "state_id": state_id,
        );
    }

    fn stop_engine(&self) -> bool {
        call_pyhandler_ret!(self, "stop_engine",)
    }

    fn stop_state(&self, _state_id: usize) -> bool {
        call_pyhandler_ret!(self, "stop_state",)
    }

    fn finalize(&mut self) {
        call_pyhandler_noret!(self, "finalize",)
    }
}

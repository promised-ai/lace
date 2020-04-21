use crate::cc::transition::StateTransition;
use crate::cc::State;
use serde::{Deserialize, Serialize};
use std::convert::Into;
use std::path::PathBuf;

/// Configuration specifying Where to save a state with given id
#[derive(
    Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash,
)]
pub struct StateOutputInfo {
    /// path to braidfile
    pub path: PathBuf,
    /// id of this state. State will be saved as `<path>/<id>.state`
    pub id: usize,
}

impl StateOutputInfo {
    pub fn new<P: Into<PathBuf>>(path: P, id: usize) -> Self {
        StateOutputInfo {
            path: path.into(),
            id,
        }
    }
}

/// Configuration for `State.update`
///
/// Sets the number of iterations, timeout, assignment algorithms, output, and
/// transitions.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct StateUpdateConfig {
    /// Maximum number of iterations to run.
    pub n_iters: usize,
    /// Timeout in seconds.
    pub timeout: Option<u64>,
    /// If `Some`, save the state immediately after running
    pub output_info: Option<StateOutputInfo>,
    /// Which transitions to run
    pub transitions: Vec<StateTransition>,
}

impl Default for StateUpdateConfig {
    fn default() -> Self {
        StateUpdateConfig {
            n_iters: 1,
            timeout: None,
            output_info: None,
            transitions: State::default_transitions(),
        }
    }
}

impl StateUpdateConfig {
    pub fn new() -> Self {
        StateUpdateConfig::default()
    }

    // Check whether we've exceeded the allotted time
    fn check_over_time(&self, duration: u64) -> bool {
        match self.timeout {
            Some(timeout) => timeout < duration,
            None => true,
        }
    }

    // Check whether we've exceeded the allotted number of iterations
    fn check_over_iters(&self, iter: usize) -> bool {
        iter > self.n_iters
    }

    /// Returns whether the run has completed by checking whether the `duration`
    /// (in seconds) the state has run is greater than `timeout` *or* the
    /// current `iter` is greater than or equal to `n_iter`
    #[allow(clippy::collapsible_if)]
    pub fn check_complete(&self, duration: u64, iter: usize) -> bool {
        let overtime = self.check_over_time(duration);
        let overiter = self.check_over_iters(iter);

        if self.timeout.is_some() {
            overtime || overiter
        } else {
            overiter
        }
    }
}

/// Configuration for `Engine.update`
///
/// Sets the number of iterations, timeout, assignment algorithms, output, and
/// transitions.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct EngineUpdateConfig {
    /// Maximum number of iterations to run.
    pub n_iters: usize,
    /// Timeout in seconds.
    #[serde(default)]
    pub timeout: Option<u64>,
    /// path to braidfile. If defined, will save states to this directory after
    /// the run.
    #[serde(default)]
    pub save_path: Option<String>,
    /// Which transitions to run
    pub transitions: Vec<StateTransition>,
}

impl Default for EngineUpdateConfig {
    fn default() -> Self {
        EngineUpdateConfig {
            n_iters: 1,
            timeout: None,
            transitions: State::default_transitions(),
            save_path: None,
        }
    }
}

impl EngineUpdateConfig {
    pub fn new() -> Self {
        EngineUpdateConfig::default()
    }

    /// Create a `StateUpdateConfig` for the state with `id`
    pub fn state_config(&self, id: usize) -> StateUpdateConfig {
        let output_info = match self.save_path {
            Some(ref path) => {
                let info = StateOutputInfo {
                    path: PathBuf::from(path),
                    id,
                };
                Some(info)
            }
            None => None,
        };

        StateUpdateConfig {
            n_iters: self.n_iters,
            timeout: self.timeout,
            transitions: self.transitions.clone(),
            output_info,
        }
    }
}

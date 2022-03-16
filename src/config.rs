use braid_cc::config::StateUpdateConfig;
use braid_cc::transition::{StateTransition, DEFAULT_STATE_TRANSITIONS};
use serde::{Deserialize, Serialize};

/// Configuration for `Engine.update`
///
/// Sets the number of iterations, timeout, assignment algorithms, output, and
/// transitions.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
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

impl EngineUpdateConfig {
    pub fn new() -> Self {
        Self {
            n_iters: 1,
            timeout: None,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
            save_path: None,
        }
    }

    /// Emit a `StateUpdateConfig` with the same settings
    pub fn state_config(&self) -> StateUpdateConfig {
        StateUpdateConfig {
            n_iters: self.n_iters,
            timeout: self.timeout,
            transitions: self.transitions.clone(),
        }
    }
}

impl Default for EngineUpdateConfig {
    fn default() -> Self {
        Self::new()
    }
}

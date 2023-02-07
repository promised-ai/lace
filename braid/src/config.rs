use crate::cc::config::StateUpdateConfig;
use crate::cc::transition::{StateTransition, DEFAULT_STATE_TRANSITIONS};
use crate::metadata::SaveConfig;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct SaveEngineConfig {
    pub path: std::path::PathBuf,
    pub save_config: SaveConfig,
}

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
    /// path to lacefile. If defined, will save states to this directory after
    /// the run or at checkpoints
    #[serde(default)]
    pub save_config: Option<SaveEngineConfig>,
    /// Which transitions to run
    pub transitions: Vec<StateTransition>,
    /// Number of iterations after which each state should be saved
    #[serde(default)]
    pub checkpoint: Option<usize>,
}

impl EngineUpdateConfig {
    pub fn new() -> Self {
        Self {
            n_iters: 1,
            timeout: None,
            transitions: Vec::new(),
            save_config: None,
            checkpoint: None,
        }
    }

    pub fn with_default_transitions() -> Self {
        Self::new().default_transitions()
    }

    pub fn default_transitions(mut self) -> Self {
        self.transitions = DEFAULT_STATE_TRANSITIONS.into();
        self
    }

    pub fn transitions(mut self, transitions: Vec<StateTransition>) -> Self {
        self.transitions.extend(transitions);
        self
    }

    pub fn transition(mut self, transition: StateTransition) -> Self {
        self.transitions.push(transition);
        self
    }

    /// Emit a `StateUpdateConfig` with the same settings
    pub fn state_config(&self) -> StateUpdateConfig {
        StateUpdateConfig {
            n_iters: self.n_iters,
            timeout: self.timeout,
            transitions: self.transitions.clone(),
        }
    }

    pub fn n_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = n_iters;
        self
    }

    pub fn checkpoint(mut self, checkpoint: Option<usize>) -> Self {
        self.checkpoint = checkpoint;
        self
    }

    pub fn timeout(mut self, seconds: Option<u64>) -> Self {
        self.timeout = seconds;
        self
    }
}

impl Default for EngineUpdateConfig {
    fn default() -> Self {
        Self::new()
    }
}

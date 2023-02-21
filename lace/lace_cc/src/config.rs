use crate::transition::{StateTransition, DEFAULT_STATE_TRANSITIONS};
use serde::{Deserialize, Serialize};
use std::convert::Into;

/// Configuration for `State.update`
///
/// Sets the number of iterations, timeout, assignment algorithms, output, and
/// transitions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StateUpdateConfig {
    /// Maximum number of iterations to run.
    pub n_iters: usize,
    /// Which transitions to run
    pub transitions: Vec<StateTransition>,
}

impl StateUpdateConfig {
    pub fn new() -> Self {
        StateUpdateConfig {
            n_iters: 1,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
        }
    }

    // Check whether we've exceeded the allotted number of iterations
    pub fn check_over_iters(&self, iter: usize) -> bool {
        iter > self.n_iters
    }
}

impl Default for StateUpdateConfig {
    fn default() -> Self {
        StateUpdateConfig::new()
    }
}

// TODO: tests

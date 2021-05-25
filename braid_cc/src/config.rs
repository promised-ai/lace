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
    /// Timeout in seconds.
    pub timeout: Option<u64>,
    /// Which transitions to run
    pub transitions: Vec<StateTransition>,
}

impl StateUpdateConfig {
    pub fn new() -> Self {
        StateUpdateConfig {
            n_iters: 1,
            timeout: None,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
        }
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

impl Default for StateUpdateConfig {
    fn default() -> Self {
        StateUpdateConfig::new()
    }
}

// TODO: tests

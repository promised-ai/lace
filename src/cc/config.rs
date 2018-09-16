use cc::transition::StateTransition;
use cc::{ColAssignAlg, RowAssignAlg};

/// Where to save this state (which has `id`)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StateOutputInfo {
    /// path to braidfile
    pub path: String,
    /// id of this state. State will be saved as `<path>/<id>.state`
    pub id: usize,
}

impl StateOutputInfo {
    pub fn new(path: String, id: usize) -> Self {
        StateOutputInfo { path, id }
    }
}

/// Configuration for `State.update`
///
/// Sets the number of iterations, timeout, assignment algorithms, output, and
/// transitions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StateUpdateConfig {
    /// Maximum number of iterations to run.
    pub n_iters: Option<usize>,
    /// Timeout in seconds.
    pub timeout: Option<u64>,
    /// Which row assignment transition kernel to use
    pub row_asgn_alg: Option<RowAssignAlg>,
    /// Which column assignment transition kernel to use
    pub col_asgn_alg: Option<ColAssignAlg>,
    /// If `Some`, save the state immediately after running
    pub output_info: Option<StateOutputInfo>,
    /// Which transitions to run
    pub transitions: Option<Vec<StateTransition>>,
}

impl Default for StateUpdateConfig {
    fn default() -> Self {
        StateUpdateConfig {
            n_iters: Some(1),
            timeout: None,
            row_asgn_alg: Some(RowAssignAlg::Slice),
            col_asgn_alg: Some(ColAssignAlg::Slice),
            output_info: None,
            transitions: None,
        }
    }
}

impl StateUpdateConfig {
    pub fn new() -> Self {
        StateUpdateConfig::default()
    }

    /// Do `run` with `n_iters` updates
    pub fn with_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = Some(n_iters);
        self
    }

    /// Run for `timeout_sec` seconds
    pub fn with_timeout(mut self, timeout_sec: u64) -> Self {
        self.timeout = Some(timeout_sec);
        self
    }

    /// Sets the row reassignment algorithm
    pub fn with_row_alg(mut self, row_asgn_alg: RowAssignAlg) -> Self {
        self.row_asgn_alg = Some(row_asgn_alg);
        self
    }

    /// Sets the column reassignment algorithm
    pub fn with_col_alg(mut self, col_asgn_alg: ColAssignAlg) -> Self {
        self.col_asgn_alg = Some(col_asgn_alg);
        self
    }

    /// After the `update` is complete, save the state
    pub fn with_output(mut self, output_info: StateOutputInfo) -> Self {
        self.output_info = Some(output_info);
        self
    }

    /// Update with the provided transitions
    pub fn with_transitions(
        mut self,
        transitions: Vec<StateTransition>,
    ) -> Self {
        self.transitions = Some(transitions);
        self
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
        match self.n_iters {
            Some(n_iters) => iter >= n_iters,
            None => true,
        }
    }

    /// Returns whether the run has completed by checking whether the `duration`
    /// (in seconds) the state has run is greater than `timeout` *or* the
    /// current `iter` is greater than or equal to `n_iter`
    pub fn check_complete(&self, duration: u64, iter: usize) -> bool {
        let overtime = self.check_over_time(duration);
        let overiter = self.check_over_iters(iter);

        if self.timeout.is_some() {
            if self.n_iters.is_some() {
                overtime || overiter
            } else {
                overtime
            }
        } else {
            if self.n_iters.is_some() {
                overiter
            } else {
                true
            }
        }
    }
}

/// Configuration for `Engine.update`
///
/// Sets the number of iterations, timeout, assignment algorithms, output, and
/// transitions.
#[derive(Serialize, Deserialize, Clone)]
pub struct EngineUpdateConfig {
    /// Maximum number of iterations to run.
    pub n_iters: Option<usize>,
    /// Timeout in seconds.
    pub timeout: Option<u64>,
    /// Which row assignment transition kernel to use
    pub row_asgn_alg: Option<RowAssignAlg>,
    /// Which column assignment transition kernel to use
    pub col_asgn_alg: Option<ColAssignAlg>,
    /// path to braidfile. If defined, will save states to this directory after
    /// the run.
    pub save_path: Option<String>,
    /// Which transitions to run
    pub transitions: Option<Vec<StateTransition>>,
}

impl Default for EngineUpdateConfig {
    fn default() -> Self {
        EngineUpdateConfig {
            n_iters: Some(1),
            timeout: None,
            row_asgn_alg: Some(RowAssignAlg::Slice),
            col_asgn_alg: Some(ColAssignAlg::Slice),
            transitions: None,
            save_path: None,
        }
    }
}

impl EngineUpdateConfig {
    pub fn new() -> Self {
        EngineUpdateConfig::default()
    }

    /// Run for a given number of iterations
    pub fn with_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = Some(n_iters);
        self
    }

    /// Run for a given number of seconds
    pub fn with_timeout(mut self, timeout_sec: u64) -> Self {
        self.timeout = Some(timeout_sec);
        self
    }

    /// Use a specific row reassignment algorithm
    pub fn with_row_alg(mut self, row_asgn_alg: RowAssignAlg) -> Self {
        self.row_asgn_alg = Some(row_asgn_alg);
        self
    }

    /// Use a specific column reassignment algorithm
    pub fn with_col_alg(mut self, col_asgn_alg: ColAssignAlg) -> Self {
        self.col_asgn_alg = Some(col_asgn_alg);
        self
    }

    /// Update with e given set of transitions
    pub fn with_transitions(
        mut self,
        transitions: Vec<StateTransition>,
    ) -> Self {
        self.transitions = Some(transitions);
        self
    }

    // TODO: should be &str?
    /// Save states to `path` upon completion
    pub fn with_path(mut self, path: String) -> Self {
        self.save_path = Some(path);
        self
    }

    /// Create a `StateUpdateConfig` for the state with `id`
    pub fn gen_state_config(&self, id: usize) -> StateUpdateConfig {
        let output_info = match self.save_path {
            Some(ref path) => {
                let info = StateOutputInfo {
                    path: path.clone(),
                    id: id,
                };
                Some(info)
            }
            None => None,
        };

        StateUpdateConfig {
            n_iters: self.n_iters.clone(),
            timeout: self.timeout.clone(),
            row_asgn_alg: self.row_asgn_alg.clone(),
            col_asgn_alg: self.col_asgn_alg.clone(),
            transitions: self.transitions.clone(),
            output_info,
        }
    }
}

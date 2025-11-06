//! Utilities for running benchmarks on real and procedural data
//!
//! # Examples
//!
//! Run a benchmark on procedural data
//!
//! ```
//! use lace::bencher::Bencher;
//! use lace::codebook::ColType;
//! use lace::cc::alg::{ColAssignAlg, RowAssignAlg};
//! use lace::cc::state::Builder;
//!
//! let state_builder = Builder::new()
//!     .n_cats(2)
//!     .n_views(2)
//!     .n_rows(100)
//!     .column_configs(20, ColType::Continuous { hyper: None, prior: None });
//!
//! let mut bencher = Bencher::from_builder(state_builder)
//!     .n_iters(1)
//!     .n_runs(3)
//!     .col_assign_alg(ColAssignAlg::Gibbs)
//!     .row_assign_alg(RowAssignAlg::FiniteCpu);
//!
//! let mut rng = rand::rng();
//! let res = bencher.run(&mut rng);
//!
//! assert_eq!(res.len(), 3);
//! ```

use std::path::PathBuf;

use rand::Rng;
use serde::Serialize;
use thiserror::Error;

use crate::cc::alg::ColAssignAlg;
use crate::cc::alg::RowAssignAlg;
use crate::cc::config::StateUpdateConfig;
use crate::cc::state::BuildStateError;
use crate::cc::state::Builder;
use crate::cc::state::State;
use crate::cc::transition::StateTransition;
use crate::codebook::Codebook;
use crate::defaults;

/// Different ways to set up a benchmarker
#[derive(Debug, Clone)]
enum BencherSetup {
    /// Benchmark on a csv
    Csv {
        codebook: Box<Codebook>,
        path: PathBuf,
    },
    /// Bencmark on a dummy state
    Builder(Builder),
}

#[derive(Debug, Error)]
pub enum GenerateStateError {
    #[error("error parsing csv: {0}")]
    Parse(#[from] crate::error::DataParseError),
    #[error("csv error: {0}")]
    Read(#[from] crate::codebook::ReadError),
    #[error("error building state: {0}")]
    BuildState(#[from] BuildStateError),
}

fn emit_prior_process<R: rand::Rng>(
    prior_process: crate::codebook::PriorProcess,
    rng: &mut R,
) -> crate::stats::prior_process::Process {
    use crate::stats::prior_process::Dirichlet;
    use crate::stats::prior_process::PitmanYor;
    use crate::stats::prior_process::Process;
    match prior_process {
        crate::codebook::PriorProcess::Dirichlet { alpha_prior } => {
            let inner = Dirichlet::from_prior(alpha_prior, rng);
            Process::Dirichlet(inner)
        }
        crate::codebook::PriorProcess::PitmanYor {
            alpha_prior,
            d_prior,
        } => {
            let inner = PitmanYor::from_prior(alpha_prior, d_prior, rng);
            Process::PitmanYor(inner)
        }
    }
}

impl BencherSetup {
    fn gen_state(
        &mut self,
        mut rng: &mut impl Rng,
    ) -> Result<State, GenerateStateError> {
        match self {
            BencherSetup::Csv {
                ref mut codebook,
                path,
            } => crate::codebook::data::read_csv(path)
                .map_err(GenerateStateError::Read)
                .and_then(|df| {
                    let state_prior_process = {
                        let prior_process = codebook
                            .state_prior_process
                            .clone()
                            .unwrap_or_default();
                        emit_prior_process(prior_process, rng)
                    };

                    let view_prior_process = {
                        let prior_process = codebook
                            .view_prior_process
                            .clone()
                            .unwrap_or_default();
                        emit_prior_process(prior_process, rng)
                    };

                    let mut codebook_tmp = Box::<Codebook>::default();

                    // swap codebook into something we can take ownership of
                    std::mem::swap(codebook, &mut codebook_tmp);
                    crate::data::df_to_col_models(*codebook_tmp, df, &mut rng)
                        .map(|(cb, features)| {
                            // put the codeboko back where it should go
                            std::mem::swap(codebook, &mut Box::new(cb));

                            State::from_prior(
                                features,
                                state_prior_process,
                                view_prior_process,
                                &mut rng,
                            )
                        })
                        .map_err(GenerateStateError::Parse)
                }),
            BencherSetup::Builder(state_builder) => state_builder
                .clone()
                .seed_from_u64(rng.next_u64())
                .build()
                .map_err(GenerateStateError::BuildState),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BencherResult {
    pub time_sec: Vec<f64>,
    pub score: Vec<f64>,
}

/// Runs inference benchmarks
#[derive(Debug, Clone)]
pub struct Bencher {
    setup: BencherSetup,
    n_runs: usize,
    n_iters: usize,
    col_asgn_alg: ColAssignAlg,
    row_asgn_alg: RowAssignAlg,
    config: Option<StateUpdateConfig>,
}

impl Bencher {
    /// Benchmark on csv data
    #[must_use]
    pub fn from_csv(codebook: Codebook, path: PathBuf) -> Self {
        Self {
            setup: BencherSetup::Csv {
                codebook: Box::new(codebook),
                path,
            },
            n_runs: 1,
            n_iters: 100,
            col_asgn_alg: defaults::COL_ASSIGN_ALG,
            row_asgn_alg: defaults::ROW_ASSIGN_ALG,
            config: None,
        }
    }

    /// Benchmark on procedurally generated States
    #[must_use]
    pub fn from_builder(state_builder: Builder) -> Self {
        Self {
            setup: BencherSetup::Builder(state_builder),
            n_runs: 1,
            n_iters: 100,
            col_asgn_alg: defaults::COL_ASSIGN_ALG,
            row_asgn_alg: defaults::ROW_ASSIGN_ALG,
            config: None,
        }
    }

    /// Repeat the benchmark a number of times
    #[must_use]
    pub fn n_runs(mut self, n_runs: usize) -> Self {
        self.n_runs = n_runs;
        self
    }

    /// Run each benchmark with a given number of inference steps
    #[must_use]
    pub fn n_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = n_iters;
        self
    }

    /// Select the row reassignment algorithm
    #[must_use]
    pub fn row_assign_alg(mut self, alg: RowAssignAlg) -> Self {
        self.row_asgn_alg = alg;
        self
    }

    /// Select the column reassignment algorithm
    #[must_use]
    pub fn col_assign_alg(mut self, alg: ColAssignAlg) -> Self {
        self.col_asgn_alg = alg;
        self
    }

    /// Provide a configuration for how the state is updated. If you only want
    /// to benchmark certain transitions, provide a config with only those
    /// transitions.
    ///
    /// The column and row reassignment algorithms in the config will override
    /// whatever is currently set
    #[must_use]
    pub fn update_config(mut self, config: StateUpdateConfig) -> Self {
        config.transitions.iter().for_each(|&t| {
            if let StateTransition::ColumnAssignment(alg) = t {
                self.col_asgn_alg = alg;
            }
        });

        config.transitions.iter().for_each(|&t| {
            if let StateTransition::RowAssignment(alg) = t {
                self.row_asgn_alg = alg;
            }
        });

        self.config = Some(config);
        self
    }

    fn state_config(&self) -> StateUpdateConfig {
        let mut config = match self.config {
            Some(ref config) => config.clone(),
            None => StateUpdateConfig {
                n_iters: 1,
                ..Default::default()
            },
        };

        let transitions = config
            .transitions
            .iter()
            .map(|t| match t {
                StateTransition::RowAssignment(_) => {
                    StateTransition::RowAssignment(self.row_asgn_alg)
                }
                StateTransition::ColumnAssignment(_) => {
                    StateTransition::ColumnAssignment(self.col_asgn_alg)
                }
                _ => *t,
            })
            .collect::<Vec<_>>();

        config.transitions = transitions;
        config
    }

    /// Run one benchmark now
    pub fn run_once(&mut self, mut rng: &mut impl Rng) -> BencherResult {
        use std::time::Instant;
        let mut state: State = self.setup.gen_state(&mut rng).unwrap();
        let config = self.state_config();
        let time_sec: Vec<f64> = (0..self.n_iters)
            .map(|_| {
                let start = Instant::now();
                state.update(config.clone(), &mut rng);
                start.elapsed().as_secs_f64()
            })
            .collect();

        BencherResult {
            time_sec,
            score: state.diagnostics.loglike,
        }
    }

    /// Run all the requested benchmarks now
    pub fn run(&mut self, mut rng: &mut impl Rng) -> Vec<BencherResult> {
        (0..self.n_runs).map(|_| self.run_once(&mut rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::ColType;

    fn quick_bencher() -> Bencher {
        let builder = Builder::new()
            .column_configs(
                5,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_rows(50);
        Bencher::from_builder(builder).n_runs(5).n_iters(17)
    }

    #[test]
    fn bencher_from_state_builder_should_return_properly_sized_result() {
        let mut bencher = quick_bencher();
        let mut rng = rand::rng();
        let results = bencher.run(&mut rng);
        assert_eq!(results.len(), 5);

        assert_eq!(results[0].time_sec.len(), 17);
        assert_eq!(results[1].time_sec.len(), 17);
        assert_eq!(results[2].time_sec.len(), 17);
        assert_eq!(results[3].time_sec.len(), 17);
        assert_eq!(results[4].time_sec.len(), 17);

        assert_eq!(results[0].score.len(), 17);
        assert_eq!(results[1].score.len(), 17);
        assert_eq!(results[2].score.len(), 17);
        assert_eq!(results[3].score.len(), 17);
        assert_eq!(results[4].score.len(), 17);
    }
}

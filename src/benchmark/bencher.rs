//! Utilities for running benchmarks on real and procedural data
//!
//! # Examples
//!
//! Run a benchmark on procedural data
//!
//! ```
//! use braid::benchmark::{Bencher, StateBuilder};
//! use braid_codebook::ColType;
//! use braid::cc::{ColAssignAlg, RowAssignAlg};
//!
//! let state_builder = StateBuilder::new()
//!     .with_cats(2)
//!     .with_views(2)
//!     .with_rows(100)
//!     .add_column_configs(20, ColType::Continuous { hyper: None });
//!
//! let bencher = Bencher::from_builder(state_builder)
//!     .with_n_iters(1)
//!     .with_n_runs(3)
//!     .with_col_assign_alg(ColAssignAlg::Gibbs)
//!     .with_row_assign_alg(RowAssignAlg::FiniteCpu);
//!
//! let mut rng = rand::thread_rng();
//! let res = bencher.run(&mut rng);
//!
//! assert_eq!(res.len(), 3);
//! ```

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::state_builder::StateBuilder;
use braid_codebook::Codebook;
use csv::ReaderBuilder;
use rand::Rng;
use serde::Serialize;

use crate::benchmark::BuildStateError;
use crate::cc::config::StateUpdateConfig;
use crate::cc::{ColAssignAlg, RowAssignAlg, State};
use crate::data::csv as braid_csv;
use crate::data::CsvParseError;
use crate::defaults;

/// Different ways to set up a benchmarker
#[derive(Debug, Clone)]
enum BencherSetup {
    /// Benchmark on a csv
    Csv { codebook: Codebook, path: PathBuf },
    /// Bencmark on a dummy state
    Builder(StateBuilder),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenerateStateError {
    CsvParseError(CsvParseError),
    DataNotFoundError,
    BuildStateError(BuildStateError),
}

impl BencherSetup {
    fn gen_state(
        &self,
        mut rng: &mut impl Rng,
    ) -> Result<State, GenerateStateError> {
        match self {
            BencherSetup::Csv { codebook, path } => ReaderBuilder::new()
                .has_headers(true)
                .from_path(Path::new(&path))
                .map_err(|_| GenerateStateError::DataNotFoundError)
                .and_then(|reader| {
                    let state_alpha_prior =
                        codebook.state_alpha_prior.clone().unwrap_or_else(
                            || braid_consts::state_alpha_prior().into(),
                        );

                    let view_alpha_prior =
                        codebook.view_alpha_prior.clone().unwrap_or_else(
                            || braid_consts::view_alpha_prior().into(),
                        );

                    braid_csv::read_cols(reader, &codebook)
                        .map(|features| {
                            State::from_prior(
                                features,
                                state_alpha_prior,
                                view_alpha_prior,
                                &mut rng,
                            )
                        })
                        .map_err(GenerateStateError::CsvParseError)
                }),
            BencherSetup::Builder(state_builder) => state_builder
                .clone()
                .with_seed(rng.next_u64())
                .build()
                .map_err(GenerateStateError::BuildStateError),
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
    pub fn from_csv(codebook: Codebook, path: PathBuf) -> Self {
        Bencher {
            setup: BencherSetup::Csv { codebook, path },
            n_runs: 1,
            n_iters: 100,
            col_asgn_alg: defaults::COL_ASSIGN_ALG,
            row_asgn_alg: defaults::ROW_ASSIGN_ALG,
            config: None,
        }
    }

    /// Benchmark on procedurally generated States
    pub fn from_builder(state_builder: StateBuilder) -> Self {
        Bencher {
            setup: BencherSetup::Builder(state_builder),
            n_runs: 1,
            n_iters: 100,
            col_asgn_alg: defaults::COL_ASSIGN_ALG,
            row_asgn_alg: defaults::ROW_ASSIGN_ALG,
            config: None,
        }
    }

    /// Repeat the benchmark a number of times
    pub fn with_n_runs(mut self, n_runs: usize) -> Self {
        self.n_runs = n_runs;
        self
    }

    /// Run each benchmark with a given number of inference steps
    pub fn with_n_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = n_iters;
        self
    }

    /// Select the row reassignment algorithm
    pub fn with_row_assign_alg(mut self, alg: RowAssignAlg) -> Self {
        self.row_asgn_alg = alg;
        self
    }

    /// Select the column reassignment algorithm
    pub fn with_col_assign_alg(mut self, alg: ColAssignAlg) -> Self {
        self.col_asgn_alg = alg;
        self
    }

    /// Select how the state is run
    pub fn with_update_config(mut self, config: StateUpdateConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Run one benchmark now
    pub fn run_once(&self, mut rng: &mut impl Rng) -> BencherResult {
        let mut state: State = self.setup.gen_state(&mut rng).unwrap();
        let config = self
            .config
            .clone()
            .unwrap_or_else(|| StateUpdateConfig::new().with_iters(1))
            .with_col_alg(self.col_asgn_alg)
            .with_row_alg(self.row_asgn_alg);

        let time_sec: Vec<f64> = (0..self.n_iters)
            .map(|_| {
                let start = SystemTime::now();
                state.update(config.clone(), &mut rng);
                let duration = start.elapsed().unwrap();

                let secs = duration.as_secs() as f64;
                let nanos = f64::from(duration.subsec_nanos()) * 1e-9;
                secs + nanos
            })
            .collect();

        BencherResult {
            time_sec,
            score: state.diagnostics.loglike,
        }
    }

    /// Run all the requested benchmarks now
    pub fn run(&self, mut rng: &mut impl Rng) -> Vec<BencherResult> {
        (0..self.n_runs).map(|_| self.run_once(&mut rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use braid_codebook::ColType;

    fn quick_bencher() -> Bencher {
        let builder = StateBuilder::new()
            .add_column_configs(5, ColType::Continuous { hyper: None })
            .with_rows(50);
        Bencher::from_builder(builder)
            .with_n_runs(5)
            .with_n_iters(17)
    }

    #[test]
    fn bencher_from_state_builder_should_return_properly_sized_result() {
        let bencher = quick_bencher();
        let mut rng = rand::thread_rng();
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

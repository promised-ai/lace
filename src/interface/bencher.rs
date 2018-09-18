extern crate csv;
extern crate rand;
extern crate rv;

use std::io;
use std::path::Path;
use std::time::SystemTime;

use self::csv::ReaderBuilder;
use self::rand::Rng;
use self::rv::dist::Gamma;

use cc::config::StateUpdateConfig;
use cc::{
    Codebook, ColAssignAlg, RowAssignAlg, State, DEFAULT_COL_ASSIGN_ALG,
    DEFAULT_ROW_ASSIGN_ALG,
};
use data::csv as braid_csv;
use data::StateBuilder;

pub enum BencherRig {
    Csv(Codebook, String),
    Builder(StateBuilder),
}

impl BencherRig {
    fn gen_state(&self, mut rng: &mut impl Rng) -> io::Result<State> {
        match self {
            BencherRig::Csv(codebook, path_string) => {
                let mut reader = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(Path::new(&path_string))?;
                let state_alpha_prior = codebook
                    .state_alpha_prior
                    .clone()
                    .unwrap_or(Gamma::new(1.0, 1.0).unwrap());
                let view_alpha_prior = codebook
                    .view_alpha_prior
                    .clone()
                    .unwrap_or(Gamma::new(1.0, 1.0).unwrap());
                let features = braid_csv::read_cols(reader, &codebook);
                let state = State::from_prior(
                    features,
                    state_alpha_prior,
                    view_alpha_prior,
                    &mut rng,
                );
                Ok(state)
            }
            BencherRig::Builder(state_builder) => state_builder.build(&mut rng),
        }
    }
}

#[derive(Serialize)]
pub struct BencherResult {
    pub time_sec: Vec<f64>,
    pub score: Vec<f64>,
}

pub struct Bencher {
    pub rig: BencherRig,
    pub n_runs: usize,
    pub n_iters: usize,
    pub col_asgn_alg: ColAssignAlg,
    pub row_asgn_alg: RowAssignAlg,
}

impl Bencher {
    pub fn from_csv(codebook: Codebook, path_string: String) -> Self {
        Bencher {
            rig: BencherRig::Csv(codebook, path_string),
            n_runs: 1,
            n_iters: 100,
            col_asgn_alg: DEFAULT_COL_ASSIGN_ALG,
            row_asgn_alg: DEFAULT_ROW_ASSIGN_ALG,
        }
    }

    // pub fn from_state(state: State) -> Self {
    //     unimplemented!();
    // }

    pub fn from_builder(state_builder: StateBuilder) -> Self {
        Bencher {
            rig: BencherRig::Builder(state_builder),
            n_runs: 1,
            n_iters: 100,
            col_asgn_alg: DEFAULT_COL_ASSIGN_ALG,
            row_asgn_alg: DEFAULT_ROW_ASSIGN_ALG,
        }
    }

    pub fn with_n_runs(mut self, n_runs: usize) -> Self {
        self.n_runs = n_runs;
        self
    }

    pub fn with_n_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = n_iters;
        self
    }

    pub fn with_row_assign_alg(mut self, alg: RowAssignAlg) -> Self {
        self.row_asgn_alg = alg;
        self
    }

    pub fn with_col_assign_alg(mut self, alg: ColAssignAlg) -> Self {
        self.col_asgn_alg = alg;
        self
    }

    pub fn run_once(&self, mut rng: &mut impl Rng) -> BencherResult {
        let mut state: State = self.rig.gen_state(&mut rng).unwrap();
        let time_sec: Vec<f64> = (0..self.n_iters)
            .map(|_| {
                let config = StateUpdateConfig::new()
                    .with_col_alg(self.col_asgn_alg)
                    .with_row_alg(self.row_asgn_alg)
                    .with_iters(1);

                let start = SystemTime::now();
                state.update(config, &mut rng);
                let duration = start.elapsed().unwrap();

                let secs = duration.as_secs() as f64;
                let nanos = duration.subsec_nanos() as f64 * 1e-9;
                secs + nanos
            }).collect();

        BencherResult {
            time_sec,
            score: state.diagnostics.loglike,
        }
    }

    pub fn run(&self, mut rng: &mut impl Rng) -> Vec<BencherResult> {
        (0..self.n_runs).map(|_| self.run_once(&mut rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc::codebook::ColType;

    fn quick_bencher() -> Bencher {
        let builder = StateBuilder::new()
            .add_columns(5, ColType::Continuous { hyper: None })
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

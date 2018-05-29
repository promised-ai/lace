extern crate csv;
extern crate rand;

use std::io;
use std::time::SystemTime;
use std::path::Path;

use self::rand::Rng;
use self::csv::ReaderBuilder;

use cc::{Codebook, State};
use cc::state::ColAssignAlg;
use cc::view::RowAssignAlg;
use data::StateBuilder;
use data::csv as braid_csv;


pub enum BencherRig {
    Csv(Codebook, String),
    Builder(StateBuilder)
}

impl BencherRig {
    fn gen_state(&self, mut rng: &mut Rng) -> io::Result<State> {
        match self {
            BencherRig::Csv(codebook, path_string) => {
                let mut reader = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(Path::new(&path_string))?;
                let state_alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
                let features = braid_csv::read_cols(reader, &codebook);
                let state = State::from_prior(features, state_alpha, &mut rng);
                Ok(state)
            },
            BencherRig::Builder(state_builder) => state_builder.build(&mut rng)
        }
    }
}

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
            col_asgn_alg: ColAssignAlg::FiniteCpu,
            row_asgn_alg: RowAssignAlg::FiniteCpu,
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
            col_asgn_alg: ColAssignAlg::FiniteCpu,
            row_asgn_alg: RowAssignAlg::FiniteCpu,
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

    pub fn run_once(&self, mut rng: &mut Rng) -> BencherResult {
        let mut state: State = self.rig.gen_state(&mut rng).unwrap();
        let time_sec: Vec<f64> = (0..self.n_iters)
            .map(|_| {
                let start = SystemTime::now();
                state.update(
                    1,
                    Some(self.row_asgn_alg),
                    Some(self.col_asgn_alg),
                    &mut rng
                );
                let duration = start.elapsed().unwrap();
                let secs = duration.as_secs() as f64;
                let nanos = duration.subsec_nanos() as f64 * 1e-9;
                secs + nanos
            }).collect();
        BencherResult {
            time_sec: time_sec,
            score: state.diagnostics.loglike
        }
    }

    pub fn run(&self, mut rng: &mut Rng) -> Vec<BencherResult> {
        (0..self.n_runs)
            .map(|_| self.run_once(&mut rng))
            .collect()
    }
}

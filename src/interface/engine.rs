extern crate csv;
extern crate indicatif;
extern crate itertools;
extern crate rand;
extern crate rusqlite;
extern crate serde_json;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::io::Result;
use std::path::Path;

use self::csv::ReaderBuilder;
use self::indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use self::rand::{FromEntropy, XorShiftRng};
use self::rusqlite::Connection;
use rayon::prelude::*;

use cc::file_utils;
use cc::state::State;
use cc::transition::StateTransition;
use cc::Codebook;
use cc::{
    ColAssignAlg, RowAssignAlg, DEFAULT_COL_ASSIGN_ALG, DEFAULT_ROW_ASSIGN_ALG,
};
use data::csv as braid_csv;
use data::{sqlite, DataSource};

pub struct Engine {
    /// Vector of states
    pub states: BTreeMap<usize, State>,
    pub codebook: Codebook,
    pub rng: XorShiftRng,
}

// TODO: more control over rng
impl Engine {
    pub fn new(
        nstates: usize,
        codebook: Codebook,
        data_source: DataSource, // TODO: add src path to enum
        id_offset: Option<usize>,
        rng_opt: Option<XorShiftRng>,
    ) -> Self {
        let mut rng = rng_opt.unwrap_or(XorShiftRng::from_entropy());
        let state_alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
        let col_models = match data_source {
            DataSource::Sqlite(..) => {
                // FIXME: Open read-only w/ flags
                let conn = Connection::open(data_source.to_path())
                    .expect("Could not open SQLite connection");
                sqlite::read_cols(&conn, &codebook)
            }
            DataSource::Csv(..) => {
                let mut reader = ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(data_source.to_path())
                    .expect("Could not open CSV");
                braid_csv::read_cols(reader, &codebook)
            }
            DataSource::Postgres(..) => unimplemented!(),
        };

        let offset = id_offset.unwrap_or(0);
        let mut states: BTreeMap<usize, State> = BTreeMap::new();
        (0..nstates).for_each(|id| {
            let features = col_models.clone();
            let state = State::from_prior(features, state_alpha, &mut rng);
            states.insert(id + offset, state);
        });
        Engine {
            states: states,
            codebook: codebook,
            rng: rng,
        }
    }

    ///  Loads the entire contents of a .braid file
    pub fn load(dir: &str) -> Result<Self> {
        let data = file_utils::load_data(dir)?;
        let mut states = file_utils::load_states(dir)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = file_utils::load_rng(dir)?;
        states.iter_mut().for_each(|(_, state)| {
            state
                .repop_data(data.clone())
                .expect("could not repopulate data");
        });
        Ok(Engine {
            states: states,
            codebook: codebook,
            rng: rng,
        })
    }

    pub fn load_states(dir: &str, ids: Vec<usize>) -> Result<Self> {
        let data = file_utils::load_data(dir)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = file_utils::load_rng(dir)?;
        let mut states: BTreeMap<usize, State> = BTreeMap::new();
        ids.iter().for_each(|id| {
            let mut state = file_utils::load_state(dir, *id).unwrap();
            state
                .repop_data(data.clone())
                .expect("Could not repopulate data");
            states.insert(*id, state);
        });
        Ok(Engine {
            states: states,
            codebook: codebook,
            rng: rng,
        })
    }

    pub fn save(&mut self, dir: &str) -> Result<()> {
        file_utils::path_validator(&dir)?;
        println!("Attempting to save");
        let has_data = file_utils::has_data(dir)?;
        if !has_data {
            print!("Saving data to {}...", dir);
            let data = self.states.values().next().unwrap().clone_data();
            file_utils::save_data(dir, &data)?;
            println!("Done.");
        }

        let has_codebook = file_utils::has_codebook(dir)?;
        if !has_codebook {
            print!("Saving codebook to {}...", dir);
            file_utils::save_codebook(dir, &self.codebook)?;
            println!("Done.");
        }
        print!("Saving states to {}...", dir);
        file_utils::save_states(dir, &mut self.states)?;
        println!("Done.");
        file_utils::save_rng(dir, &self.rng)?;
        Ok(())
    }

    pub fn from_sqlite(
        db_path: &Path,
        codebook: Codebook,
        nstates: usize,
        id_offset: Option<usize>,
        rng_opt: Option<XorShiftRng>,
    ) -> Self {
        let mut rng = rng_opt.unwrap_or(XorShiftRng::from_entropy());
        let alpha: f64 = codebook.state_alpha().unwrap_or(1.0);
        let conn = Connection::open(&db_path).unwrap();
        let ftrs = sqlite::read_cols(&conn, &codebook);

        let offset = id_offset.unwrap_or(0);
        let mut states: BTreeMap<usize, State> = BTreeMap::new();
        (0..nstates).for_each(|id| {
            let state = State::from_prior(ftrs.clone(), alpha, &mut rng);
            states.insert(id + offset, state);
        });
        Engine {
            states: states,
            codebook: codebook,
            rng: rng,
        }
    }

    /// TODO
    pub fn from_postegres(_path: &Path) -> Self {
        unimplemented!();
    }

    /// Run each `State` in the `Engine` for `n_iters` iterations using the
    /// default algorithms and transitions. If `show_progress` is `true` then
    /// each `State` will maintain a progress bar.
    pub fn run(&mut self, n_iters: usize, show_progress: bool) {
        let row_asgn_alg = DEFAULT_ROW_ASSIGN_ALG;
        let col_asgn_alg = DEFAULT_COL_ASSIGN_ALG;
        let transitions = State::default_transitions();
        self.update(
            n_iters,
            row_asgn_alg,
            col_asgn_alg,
            transitions,
            show_progress,
        );
    }

    /// Run each `State` in the `Engine` for `n_iters` with specific
    /// algorithms and transitions.
    pub fn update(
        &mut self,
        n_iters: usize,
        row_asgn_alg: RowAssignAlg,
        col_asgn_alg: ColAssignAlg,
        transitions: Vec<StateTransition>,
        show_progress: bool,
    ) {
        let m = MultiProgress::new();
        let sty = ProgressStyle::default_bar();

        if show_progress {
            self.states.par_iter_mut().for_each(|(_, state)| {
                let pb = m.add(ProgressBar::new(n_iters as u64));
                pb.set_style(sty.clone());

                let mut rng = rand::thread_rng();
                state.update_pb(
                    n_iters,
                    Some(row_asgn_alg),
                    Some(col_asgn_alg),
                    Some(transitions.clone()),
                    &mut rng,
                    &pb,
                );
            });
            m.join_and_clear().unwrap();
        } else {
            self.states.par_iter_mut().for_each(|(_, state)| {
                let mut rng = rand::thread_rng();
                state.update(
                    n_iters,
                    Some(row_asgn_alg),
                    Some(col_asgn_alg),
                    Some(transitions.clone()),
                    &mut rng,
                );
            });
        }
    }

    /// Returns the number of stats in the `Oracle`
    pub fn nstates(&self) -> usize {
        self.states.len()
    }
}

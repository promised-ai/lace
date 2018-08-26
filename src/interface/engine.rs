extern crate csv;
extern crate indicatif;
extern crate itertools;
extern crate rand;
extern crate rusqlite;
extern crate rv;
extern crate serde_json;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::io::Result;

use self::csv::ReaderBuilder;
use self::indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use self::rand::{SeedableRng, XorShiftRng};
use self::rusqlite::Connection;
use self::rv::dist::InvGamma;
use rayon::prelude::*;

use cc::file_utils;
use cc::state::State;
use cc::transition::StateTransition;
use cc::Codebook;
use cc::{
    ColAssignAlg, ColModel, RowAssignAlg, DEFAULT_COL_ASSIGN_ALG,
    DEFAULT_ROW_ASSIGN_ALG,
};
use data::csv as braid_csv;
use data::{sqlite, DataSource};

#[derive(Clone)]
pub struct Engine {
    /// Vector of states
    pub states: BTreeMap<usize, State>,
    pub codebook: Codebook,
    pub rng: XorShiftRng,
}

fn col_models_from_data_src(
    codebook: &Codebook,
    data_source: &DataSource,
) -> Vec<ColModel> {
    match data_source {
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
    }
}

impl Engine {
    pub fn new(
        nstates: usize,
        codebook: Codebook,
        data_source: DataSource,
        id_offset: usize,
        mut rng: XorShiftRng,
    ) -> Self {
        let col_models = col_models_from_data_src(&codebook, &data_source);
        let state_alpha_prior = codebook
            .get_state_alpha_prior()
            .unwrap_or(InvGamma::new(1.0, 1.0).unwrap());
        let view_alpha_prior = codebook
            .get_view_alpha_prior()
            .unwrap_or(InvGamma::new(1.0, 1.0).unwrap());
        let mut states: BTreeMap<usize, State> = BTreeMap::new();

        (0..nstates).for_each(|id| {
            let features = col_models.clone();
            let state = State::from_prior(
                features,
                state_alpha_prior.clone(),
                view_alpha_prior.clone(),
                &mut rng,
            );
            states.insert(id + id_offset, state);
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

    /// Appends new features from a `DataSource` and a `Codebook`
    pub fn append_features(
        &mut self,
        mut codebook: Codebook,
        data_source: DataSource,
    ) {
        let id_map = self.codebook.merge_cols(&codebook);
        codebook.reindex_cols(&id_map);
        let col_models = col_models_from_data_src(&codebook, &data_source);
        let mut mrng = &mut self.rng;
        self.states.values_mut().for_each(|state| {
            state
                .insert_new_features(col_models.clone(), &mut mrng)
                .expect("Failed to insert features");
        });
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

        let mut trngs: Vec<XorShiftRng> = (0..self.nstates())
            .map(|_| XorShiftRng::from_rng(&mut self.rng).unwrap())
            .collect();

        // rayon has a hard time doing self.states.par_iter().zip(..), so we
        // grab some mutable references explicitly
        let mut states: Vec<&mut State> =
            self.states.iter_mut().map(|(_, state)| state).collect();

        if show_progress {
            states.par_iter_mut().zip(trngs.par_iter_mut()).for_each(
                |(state, mut trng)| {
                    let pb = m.add(ProgressBar::new(n_iters as u64));
                    pb.set_style(sty.clone());

                    state.update_pb(
                        n_iters,
                        Some(row_asgn_alg),
                        Some(col_asgn_alg),
                        Some(transitions.clone()),
                        &mut trng,
                        &pb,
                    );
                },
            );
            m.join_and_clear().unwrap();
        } else {
            states.par_iter_mut().zip(trngs.par_iter_mut()).for_each(
                |(state, mut trng)| {
                    state.update(
                        n_iters,
                        Some(row_asgn_alg),
                        Some(col_asgn_alg),
                        Some(transitions.clone()),
                        &mut trng,
                    );
                },
            );
        }
    }

    /// Returns the number of stats in the `Oracle`
    pub fn nstates(&self) -> usize {
        self.states.len()
    }
}

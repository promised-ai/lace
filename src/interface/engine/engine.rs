use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use braid_codebook::Codebook;
use csv::ReaderBuilder;
use log::info;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use rusqlite::Connection;

use super::error::{
    AppendFeaturesError, AppendRowsError, DataParseError, NewEngineError,
};
use crate::cc::config::EngineUpdateConfig;
use crate::cc::state::State;
use crate::cc::{file_utils, AppendRowsData, ColModel};
use crate::data::{csv as braid_csv, sqlite, DataSource};
use crate::file_config::{FileConfig, SerializedType};

/// The engine runs states in parallel
#[derive(Clone)]
pub struct Engine {
    /// Vector of states
    pub states: BTreeMap<usize, State>,
    pub codebook: Codebook,
    pub rng: Xoshiro256Plus,
}

fn col_models_from_data_src(
    codebook: &Codebook,
    data_source: &DataSource,
) -> Result<Vec<ColModel>, DataParseError> {
    match data_source {
        DataSource::Sqlite(..) => {
            // FIXME: Open read-only w/ flags
            let conn = Connection::open(Path::new(&data_source.to_string()))
                .expect("Could not open SQLite connection");
            Ok(sqlite::read_cols(&conn, &codebook))
        }
        DataSource::Csv(..) => ReaderBuilder::new()
            .has_headers(true)
            .from_path(data_source.to_os_string())
            .map_err(|_| DataParseError::IoError)
            .and_then(|reader| {
                braid_csv::read_cols(reader, &codebook)
                    .map_err(DataParseError::CsvParseError)
            }),
        DataSource::Postgres(..) => {
            Err(DataParseError::UnsupportedDataSourceError)
        }
    }
}

/// Maintains and samples states
impl Engine {
    /// Create a new engine
    ///
    /// # Arguments
    /// - nstates: number of states
    /// - id_offset: the state IDs will start at `id_offset`. This is useful
    ///   for when you run multiple engines on multiple machines and want to
    ///   easily combine the states in a single `Oracle` after the runs
    pub fn new(
        nstates: usize,
        codebook: Codebook,
        data_source: DataSource,
        id_offset: usize,
        mut rng: Xoshiro256Plus,
    ) -> Result<Self, NewEngineError> {
        if nstates == 0 {
            return Err(NewEngineError::ZeroStatesRequestedError);
        }

        let col_models = col_models_from_data_src(&codebook, &data_source)
            .map_err(NewEngineError::DataParseError)?;

        let state_alpha_prior = codebook
            .state_alpha_prior
            .clone()
            .unwrap_or_else(|| braid_consts::state_alpha_prior().into());

        let view_alpha_prior = codebook
            .view_alpha_prior
            .clone()
            .unwrap_or_else(|| braid_consts::view_alpha_prior().into());

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

        Ok(Engine {
            states,
            codebook,
            rng,
        })
    }

    /// Re-seed the RNG
    pub fn seed_from_u64(&mut self, seed: u64) {
        self.rng = Xoshiro256Plus::seed_from_u64(seed);
    }

    ///  Load a braidfile into an `Engine`.
    ///
    /// # Notes
    ///
    ///  The RNG is not saved. It is re-seeded upon load.
    pub fn load(dir: &Path) -> io::Result<Self> {
        let config = file_utils::load_file_config(dir).unwrap_or_default();
        let data = file_utils::load_data(dir, &config)?;
        let mut states = file_utils::load_states(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = Xoshiro256Plus::from_entropy();

        states
            .iter_mut()
            .for_each(|(_, state)| state.repop_data(data.clone()));

        Ok(Engine {
            states,
            codebook,
            rng,
        })
    }

    /// Load a braidfile into an `Engine` using only the `State`s with the
    /// specified IDs
    ///
    /// # Notes
    ///
    ///  The RNG is not saved. It is re-seeded upon load.
    pub fn load_states(dir: &Path, mut ids: Vec<usize>) -> io::Result<Self> {
        let config = file_utils::load_file_config(dir).unwrap_or_default();
        let data = file_utils::load_data(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = Xoshiro256Plus::from_entropy();

        let states: io::Result<BTreeMap<usize, State>> = ids
            .drain(..)
            .map(|id| {
                file_utils::load_state(dir, id, &config).map(|mut state| {
                    state.repop_data(data.clone());
                    (id, state)
                })
            })
            .collect();

        states.map(|states| Engine {
            states,
            codebook,
            rng,
        })
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.states[&0].nrows()
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.states[&0].ncols()
    }

    /// Appends new features from a `DataSource` and a `Codebook`
    ///
    /// # Arguments
    /// - partial_codebook: A codebook that contains the column metadata for
    ///   the new features.
    /// - data_source: The DataSource that points to the new features
    pub fn append_features(
        &mut self,
        mut partial_codebook: Codebook,
        data_source: DataSource,
    ) -> Result<(), AppendFeaturesError> {
        use crate::cc::Feature;
        use crate::data::CsvParseError;
        // TODO: partial_codebook should be a vector of ColMetadata
        let id_map = self
            .codebook
            .merge_cols(&partial_codebook)
            .map_err(|err| err.into())?;

        partial_codebook.reindex_cols(&id_map);

        col_models_from_data_src(&partial_codebook, &data_source)
            .map_err(|err| match err {
                DataParseError::IoError => AppendFeaturesError::IoError,
                DataParseError::CsvParseError(
                    CsvParseError::MissingCsvColumnsError,
                ) => AppendFeaturesError::CodebookDataColumnNameMismatchError,
                DataParseError::CsvParseError(
                    CsvParseError::MissingCodebookColumnsError,
                ) => AppendFeaturesError::CodebookDataColumnNameMismatchError,
                _ => AppendFeaturesError::DataParseError(err),
            })
            .and_then(|col_models| {
                if col_models.iter().any(|cm| cm.len() != self.nrows()) {
                    Err(AppendFeaturesError::ColumnLengthError)
                } else {
                    let mut mrng = &mut self.rng;
                    self.states.values_mut().for_each(|state| {
                        state
                            .insert_new_features(col_models.clone(), &mut mrng);
                    });
                    Ok(())
                }
            })
    }

    /// Appends new rows from a`DataSource`. All columns must be present in
    /// the new data.
    ///
    /// **NOTE**: Currently only csv is supported
    pub fn append_rows(
        &mut self,
        data_source: DataSource,
    ) -> Result<(), AppendRowsError> {
        let row_data = match data_source {
            DataSource::Csv(..) => ReaderBuilder::new()
                .has_headers(true)
                .from_path(data_source.to_os_string())
                .map_err(|_| AppendRowsError::IoError)
                .and_then(|reader| {
                    braid_csv::row_data_from_csv(reader, &mut self.codebook)
                        .map_err(|err| {
                            AppendRowsError::DataParseError(err.into())
                        })
                }),
            _ => Err(AppendRowsError::UnsupportedDataSourceError),
        }?;

        // This is actually stored column wise. These are the columns that
        // comprise the new rows.
        let new_rows: Vec<&AppendRowsData> = row_data.iter().collect();

        // XXX: This will be caught as a CsvParseError
        if new_rows.len() != self.states[&0].ncols() {
            return Err(AppendRowsError::RowLengthMismatchError);
        }

        // XXX: This isn't really possible given that the data are passed in
        // as a table.
        let ncols = new_rows[0].len();
        if new_rows.iter().skip(1).any(|col| col.len() != ncols) {
            return Err(AppendRowsError::ColumLengthMismatchError);
        }

        for state in self.states.values_mut() {
            state.append_rows(new_rows.clone(), &mut self.rng);
        }

        Ok(())
    }

    /// Save the Engine to a braidfile
    pub fn save_to(self, dir: &Path) -> EngineSaver {
        EngineSaver::new(self, dir)
    }

    /// Run each `State` in the `Engine` for `n_iters` iterations using the
    /// default algorithms and transitions. If `show_progress` is `true` then
    /// each `State` will maintain a progress bar.
    pub fn run(&mut self, n_iters: usize) {
        let config = EngineUpdateConfig::new().with_iters(n_iters);
        self.update(config);
    }

    /// Run each `State` in the `Engine` for `n_iters` with specific
    /// algorithms and transitions.
    pub fn update(&mut self, config: EngineUpdateConfig) {
        let mut trngs: Vec<Xoshiro256Plus> = (0..self.nstates())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng).unwrap())
            .collect();

        // rayon has a hard time doing self.states.par_iter().zip(..), so we
        // grab some mutable references explicitly
        let mut states: Vec<&mut State> =
            self.states.iter_mut().map(|(_, state)| state).collect();

        states
            .par_iter_mut()
            .zip(trngs.par_iter_mut())
            .enumerate()
            .for_each(|(id, (state, mut trng))| {
                state.update(config.gen_state_config(id), &mut trng);
            });
    }

    /// Returns the number of stats
    pub fn nstates(&self) -> usize {
        self.states.len()
    }
}

/// Object for saving `Engine` data to a given directory
pub struct EngineSaver {
    dir: PathBuf,
    engine: Engine,
    serialized_type: Option<SerializedType>,
}

/// Saves the engine
///
/// # Notes
///
/// The RNG state is not saved. It is re-seeded on load.
///
/// # Example
///
/// ```ignore
/// engine = engine
///     .save_to("path/to/engine")
///     .with_serialized_type(SerializedType::Bincode)
///     .save()
///     .unwrap()
/// ```
impl EngineSaver {
    pub fn new<P: Into<PathBuf>>(engine: Engine, dir: P) -> Self {
        EngineSaver {
            dir: dir.into(),
            engine,
            serialized_type: None,
        }
    }

    /// Which format in which to save the states and data
    pub fn with_serialized_type(
        mut self,
        serialized_type: SerializedType,
    ) -> Self {
        self.serialized_type = Some(serialized_type);
        self
    }

    /// Save the Engine to a braidfile and release the engine
    pub fn save(mut self) -> io::Result<Engine> {
        let file_config = FileConfig {
            serialized_type: self.serialized_type,
            ..FileConfig::default()
        };

        let dir = self.dir.as_path();
        let dir_str = dir.to_str().unwrap();

        file_utils::path_validator(dir)?;
        file_utils::save_file_config(dir, &file_config)?;

        let data = self.engine.states.values().next().unwrap().clone_data();
        file_utils::save_data(&dir, &data, &file_config)?;

        info!("Saving codebook to {}...", dir_str);
        file_utils::save_codebook(&dir, &self.engine.codebook)?;

        info!("Saving states to {}...", dir_str);
        file_utils::save_states(&dir, &mut self.engine.states, &file_config)?;

        info!("Done saving.");
        Ok(self.engine)
    }
}

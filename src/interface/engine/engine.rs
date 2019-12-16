use std::io;
use std::path::{Path, PathBuf};

use braid_codebook::Codebook;
use braid_stats::Datum;
use braid_utils::ForEachOk;
use csv::ReaderBuilder;
use log::info;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use rusqlite::Connection;

use super::data::{create_new_columns, insert_data_tasks, InsertMode, Row};
use super::error::{
    AppendFeaturesError, AppendRowsError, DataParseError, InsertDataError,
    NewEngineError,
};
use super::RowAlignmentStrategy;
use crate::cc::config::EngineUpdateConfig;
use crate::cc::state::State;
use crate::cc::{file_utils, AppendRowsData, ColModel};
use crate::data::{csv as braid_csv, sqlite, DataSource};
use crate::file_config::{FileConfig, SerializedType};

/// The engine runs states in parallel
#[derive(Clone)]
pub struct Engine {
    /// Vector of states
    pub states: Vec<State>,
    pub state_ids: Vec<usize>,
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

        let states: Vec<State> = (0..nstates)
            .map(|_| {
                let features = col_models.clone();
                State::from_prior(
                    features,
                    state_alpha_prior.clone(),
                    view_alpha_prior.clone(),
                    &mut rng,
                )
            })
            .collect();

        let state_ids = (id_offset..nstates + id_offset).collect();

        Ok(Engine {
            states,
            state_ids,
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
        let (mut states, state_ids) = file_utils::load_states(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = Xoshiro256Plus::from_entropy();

        states
            .iter_mut()
            .for_each(|state| state.repop_data(data.clone()));

        Ok(Engine {
            states,
            state_ids,
            codebook,
            rng,
        })
    }

    /// Load a braidfile into an `Engine` using only the `State`s with the
    /// specified IDs
    ///
    /// # Notes
    ///
    /// The RNG is not saved. It is re-seeded upon load.
    pub fn load_states(
        dir: &Path,
        mut state_ids: Vec<usize>,
    ) -> io::Result<Self> {
        let config = file_utils::load_file_config(dir).unwrap_or_default();
        let data = file_utils::load_data(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = Xoshiro256Plus::from_entropy();

        let states: io::Result<Vec<State>> = state_ids
            .drain(..)
            .map(|id| {
                file_utils::load_state(dir, id, &config).map(|mut state| {
                    state.repop_data(data.clone());
                    state
                })
            })
            .collect();

        states.map(|states| Engine {
            states,
            state_ids,
            codebook,
            rng,
        })
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.states[0].nrows()
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.states[0].ncols()
    }

    /// Insert a datum at the provided index
    fn insert_datum(
        &mut self,
        row_ix: usize,
        col_ix: usize,
        datum: Datum,
    ) -> Result<(), InsertDataError> {
        self.states.iter_mut().for_each(|state| {
            state.insert_datum(row_ix, col_ix, datum.clone());
        });
        Ok(())
    }

    // TODO: Lots of opportunity for optimization
    /// Insert new, or overwrite existing data
    ///
    /// # Notes
    /// It is assumed that the user will run a transition after the new data
    /// are inserted. No effort is made to update any of the state according to
    /// the MCMC kernel, so the state will likely be sub optimal.
    ///
    /// New columns are assigned to a random existing view; new rows are
    /// reassigned using the Gibbs kernel. Overwritten cells are left alone.
    ///
    /// # Arguments
    /// - rows: The rows of data containing the cells to insert or re-write
    /// - partial_codebook: Contains the column metadata for only the new
    ///   columns to be inserted. The columns will be inserted in the order
    ///   they appear in the column metadata. If there are columns that appear
    ///   in the column metadata that do not appear in `rows`, it will cause an
    ///   error.
    /// - mode: Defines how states may be modified.
    ///
    /// # Example
    ///
    /// Add a pegasus row with a few important values.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid_stats::Datum;
    /// use braid::{Row, Value, InsertMode, InsertOverwrite};
    ///
    /// let mut engine = Example::Animals.engine().unwrap();
    /// let starting_rows = engine.nrows();
    ///
    /// let rows = vec![
    ///     Row {
    ///         row_name: "pegasus".into(),
    ///         values: vec![
    ///             Value {
    ///                 col_name: "flys".into(),
    ///                 value: Datum::Categorical(1),
    ///             },
    ///             Value {
    ///                 col_name: "hooves".into(),
    ///                 value: Datum::Categorical(1),
    ///             },
    ///             Value {
    ///                 col_name: "swims".into(),
    ///                 value: Datum::Categorical(0),
    ///             },
    ///         ]
    ///     }
    /// ];
    ///
    /// // Allow insert_data to add new rows, but not new columns, and prevent
    /// // any existing data (even missing cells) from being overwritten.
    /// let result = engine.insert_data(
    ///     rows,
    ///     None,
    ///     InsertMode::DenyNewColumns(InsertOverwrite::Deny)
    /// );
    ///
    /// assert!(result.is_ok());
    /// assert_eq!(engine.nrows(), starting_rows + 1);
    /// ```
    ///
    /// Add a column that may help us categorize a new type of animal.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_stats::Datum;
    /// # use braid::{Row, Value, InsertMode, InsertOverwrite};
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// # let starting_rows = engine.nrows();
    /// use std::convert::TryInto;
    /// use braid_codebook::{Codebook, ColMetadata, ColType, SpecType};
    ///
    /// let rows = vec![
    ///     Row {
    ///         row_name: "bat".into(),
    ///         values: vec![
    ///             Value {
    ///                 col_name: "drinks+blood".into(),
    ///                 value: Datum::Categorical(1),
    ///             },
    ///         ]
    ///     },
    ///     Row {
    ///         row_name: "beaver".into(),
    ///         values: vec![
    ///             Value {
    ///                 col_name: "drinks+blood".into(),
    ///                 value: Datum::Categorical(0),
    ///             },
    ///         ]
    ///     }
    /// ];
    ///
    /// // The partial codebook is required to define the data type and
    /// // distribution of new columns
    /// let partial_codebook = Codebook {
    ///     table_name: "partial".into(),
    ///     state_alpha_prior: None,
    ///     view_alpha_prior: None,
    ///     col_metadata: vec![ColMetadata {
    ///         name: "drinks+blood".into(),
    ///         spec_type: SpecType::Other,
    ///         coltype: ColType::Categorical {
    ///             k: 2,
    ///             hyper: None,
    ///             value_map: None,
    ///         },
    ///         notes: None,
    ///     }]
    ///     .try_into()
    ///     .unwrap(),
    ///     comments: None,
    ///     row_names: None,
    /// };
    ///
    /// let starting_cols = engine.ncols();
    ///
    /// // Allow insert_data to add new columns, but not new rows, and prevent
    /// // any existing data (even missing cells) from being overwritten.
    /// let result = engine.insert_data(
    ///     rows,
    ///     Some(partial_codebook),
    ///     InsertMode::DenyNewRows(InsertOverwrite::Deny)
    /// );
    ///
    /// assert!(result.is_ok());
    /// assert_eq!(engine.ncols(), starting_cols + 1);
    /// ```
    pub fn insert_data(
        &mut self,
        rows: Vec<Row>,
        partial_codebook: Option<Codebook>,
        mode: InsertMode,
    ) -> Result<(), InsertDataError> {
        // TODO: Errors not caught
        // - user inserts missing data into new column so the column is all
        //   missing data, which wold probably break transitions
        // - user insert missing data into new row so that the row is all
        //   missing data. This might not break the transitions, but it is
        //   wasteful.
        // Figure out the tasks required to insert these data, and convert all
        // String row/col indices into usize.
        // TODO: insert_data_tasks should just take the rows
        let (tasks, mut ixrows) =
            insert_data_tasks(&rows, &partial_codebook, &self)?;

        // Make sure the tasks required line up with the user-defined insert
        // mode.
        tasks.validate_insert_mode(&mode)?;

        // Add empty columns to the Engine if needed
        match partial_codebook {
            // There is partial codebook and there are new columns to add
            Some(cb) if !tasks.new_cols.is_empty() => {
                tasks.new_cols.iter().for_each_ok(|col| {
                    if cb.col_metadata.contains_key(col) {
                        Ok(())
                    } else {
                        Err(
                            InsertDataError::NewColumnNotInPartialCodebookError(
                                col.clone(),
                            ),
                        )
                    }
                })?;

                if cb.col_metadata.len() > tasks.new_cols.len() {
                    // There are more columns in the partial codebook than are
                    // in the inserted data.
                    Err(InsertDataError::TooManyEntriesInPartialCodebookError)
                } else {
                    println!("Adding new colums!");
                    // create blank (data-less) columns and insert them into
                    // the States
                    let shape = (self.nrows(), self.ncols());
                    create_new_columns(
                        &cb,
                        shape,
                        &tasks.new_cols,
                        &mut self.rng,
                    )
                    .map(|col_models| {
                        // Inserts blank columns into random existing views.
                        // It is assumed that another reassignment transition
                        // will be run after the data are inserted.
                        let mut rng = &mut self.rng;
                        self.states.iter_mut().for_each(|state| {
                            state.append_blank_features(
                                col_models.clone(),
                                &mut rng,
                            );
                        });

                        // Combine the codebooks
                        // XXX: if a panic happens here its our fault.
                        self.codebook.merge_cols(cb).unwrap();
                    })
                }
            }
            // There are new columns, but no partial codebook
            None if !tasks.new_cols.is_empty() => {
                Err(InsertDataError::NoPartialCodebookError)
            }
            // Can ignore other cases (no new columns)
            _ => Ok(()),
        }?;

        // Create empty rows if needed
        if !tasks.new_rows.is_empty() {
            let n_new_rows = tasks.new_rows.len();
            self.states
                .iter_mut()
                .for_each(|state| state.extend_cols(n_new_rows));

            // Add the row names to the codebook
            // NOTE: assumes the function would have already errored if row
            // names were not in the codebook
            self.codebook.row_names.as_mut().map(|row_names| {
                tasks
                    .new_rows
                    .iter()
                    .for_each(|row_name| row_names.push(row_name.to_owned()));
            });
        }

        // Start inserting data
        ixrows.iter_mut().for_each_ok(|ixrow| {
            let row_ix = &ixrow.row_ix;
            ixrow.values.drain(..).for_each_ok(|value| {
                self.insert_datum(*row_ix, value.col_ix, value.value)
            })
        })?;

        // Find all unassigned (new) rows and re-assign them
        let mut rng = &mut self.rng;
        self.states
            .iter_mut()
            .for_each(|state| state.assign_unassigned(&mut rng));

        Ok(())
    }

    /// Appends new features from a `DataSource` and a `Codebook`
    ///
    /// # Arguments
    /// - partial_codebook: A codebook that contains the column metadata for
    ///   the new features.
    /// - data_source: The DataSource that points to the new features
    pub fn append_features(
        &mut self,
        partial_codebook: Codebook,
        data_source: DataSource,
        row_align: RowAlignmentStrategy,
    ) -> Result<(), AppendFeaturesError> {
        use crate::cc::Feature;
        use crate::data::CsvParseError;

        if row_align == RowAlignmentStrategy::CheckNames {
            match (&self.codebook.row_names, &partial_codebook.row_names) {
                (Some(names_p), Some(names_c)) => {
                    if names_p.iter().zip(names_c.iter()).any(|(c, p)| c != p) {
                        Err(AppendFeaturesError::RowNameMismatchError)
                    } else {
                        Ok(())
                    }
                }
                (None, Some(_)) => {
                    Err(AppendFeaturesError::NoRowNamesInParentError)
                }
                (Some(_), None) => {
                    Err(AppendFeaturesError::NoRowNamesInChildError)
                }
                _ => Err(AppendFeaturesError::NoRowNamesError),
            }
        } else {
            Ok(())
        }?;

        // FIXME: Currently, the user is expected to ensure the data are
        // ordered correctly. In the future, the codebook should contain row
        // names and we should reorder the data to match the existing order. We
        // should also return an error if specific rows are missing or there
        // are new rows.
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
                    self.states.iter_mut().for_each(|state| {
                        state
                            .insert_new_features(col_models.clone(), &mut mrng);
                    });
                    Ok(())
                }
            })
            .and_then(|_| {
                self.codebook
                    .merge_cols(partial_codebook)
                    .map_err(|err| err.into())
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
        if new_rows.len() != self.states[0].ncols() {
            return Err(AppendRowsError::RowLengthMismatchError);
        }

        // XXX: This isn't really possible given that the data are passed in
        // as a table.
        let ncols = new_rows[0].len();
        if new_rows.iter().skip(1).any(|col| col.len() != ncols) {
            return Err(AppendRowsError::ColumLengthMismatchError);
        }

        for state in self.states.iter_mut() {
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
        self.states
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

        let data = self.engine.states.iter().next().unwrap().clone_data();
        file_utils::save_data(&dir, &data, &file_config)?;

        info!("Saving codebook to {}...", dir_str);
        file_utils::save_codebook(&dir, &self.engine.codebook)?;

        info!("Saving states to {}...", dir_str);
        file_utils::save_states(
            &dir,
            &mut self.engine.states,
            &self.engine.state_ids,
            &file_config,
        )?;

        info!("Done saving.");
        Ok(self.engine)
    }
}

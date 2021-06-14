use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use braid_cc::feature::{ColModel, Feature};
use braid_cc::state::State;
use braid_codebook::{Codebook, ColMetadata, ColMetadataList};
use braid_data::{Datum, SummaryStatistics};
use braid_metadata::latest::Metadata;
use csv::ReaderBuilder;
use log::info;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::data::{
    append_empty_columns, insert_data_tasks, maybe_add_categories,
    AppendStrategy, InsertDataActions, Row, WriteMode,
};
use super::error::{
    DataParseError, InsertDataError, NewEngineError, RemoveDataError,
};
use crate::config::EngineUpdateConfig;
use crate::data::{csv as braid_csv, DataSource};
use crate::file_config::{FileConfig, SerializedType};
use crate::file_utils;
use crate::{HasData, HasStates, Oracle, OracleT, TableIndex};

/// The engine runs states in parallel
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "Metadata", into = "Metadata")]
pub struct Engine {
    /// Vector of states
    pub states: Vec<State>,
    pub state_ids: Vec<usize>,
    pub codebook: Codebook,
    pub rng: Xoshiro256Plus,
}

impl From<Oracle> for Engine {
    fn from(oracle: Oracle) -> Engine {
        Engine {
            state_ids: (0..oracle.states.len()).collect(),
            states: oracle.states,
            codebook: oracle.codebook,
            rng: Xoshiro256Plus::from_entropy(),
        }
    }
}

impl HasStates for Engine {
    #[inline]
    fn states(&self) -> &Vec<State> {
        &self.states
    }

    #[inline]
    fn states_mut(&mut self) -> &mut Vec<State> {
        &mut self.states
    }
}

impl HasData for Engine {
    #[inline]
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics {
        let state = &self.states[0];
        let view_ix = state.asgn.asgn[ix];
        // XXX: Cloning the data could be very slow
        state.views[view_ix].ftrs[&ix].clone_data().summarize()
    }

    #[inline]
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum {
        self.states[0].datum(row_ix, col_ix)
    }
}

impl OracleT for Engine {}

fn col_models_from_data_src<R: rand::Rng>(
    codebook: &Codebook,
    data_source: &DataSource,
    mut rng: &mut R,
) -> Result<Vec<ColModel>, DataParseError> {
    match data_source {
        DataSource::Csv(..) => {
            ReaderBuilder::new()
                .has_headers(true)
                .from_path(data_source.to_os_string().expect(
                    "This shouldn't fail since we have a Csv datasource",
                ))
                .map_err(DataParseError::CsvError)
                .and_then(|reader| {
                    braid_csv::read_cols(reader, &codebook, &mut rng)
                        .map_err(DataParseError::CsvParseError)
                })
        }
        DataSource::Postgres(..) => Err(DataParseError::UnsupportedDataSource),
        DataSource::Empty if !codebook.col_metadata.is_empty() => {
            Err(DataParseError::ColumnMetadataSuppliedForEmptyData)
        }
        DataSource::Empty if !codebook.row_names.is_empty() => {
            Err(DataParseError::RowNamesSuppliedForEmptyData)
        }
        DataSource::Empty => Ok(vec![]),
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
            return Err(NewEngineError::ZeroStatesRequested);
        }

        let col_models =
            col_models_from_data_src(&codebook, &data_source, &mut rng)
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
        use braid_metadata::latest::METADATA_VERSION;

        let config = file_utils::load_file_config(dir)?;
        if config.version > METADATA_VERSION {
            panic!(
                "{:?} was saved with metadata version {}, but this version \
                of braid only support up to version {}.",
                dir, config.version, METADATA_VERSION
            );
        }

        let data = file_utils::load_data(dir, &config)?;
        let (mut states, state_ids) = file_utils::load_states(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;
        let rng = file_utils::load_rng(dir)?;

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
        let rng = file_utils::load_rng(dir)?;

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

    /// Delete n rows starting at index ix.
    ///
    /// If ix + n exceeds the number of rows, all of the rows starting at ix
    /// will be deleted.
    pub fn del_rows_at(&mut self, ix: usize, n: usize) {
        if n == 0 {
            return;
        }

        let n_rows = self.states[0].n_rows();
        let n = if ix + n > n_rows {
            n - (ix + n - n_rows)
        } else {
            n
        };

        if n == 0 {
            return;
        }

        assert!(ix + n <= n_rows);

        self.states
            .iter_mut()
            .for_each(|state| state.del_rows_at(ix, n));

        (0..n).for_each(|_| {
            // TODO: get rid of this clone by adding a method to RowNameList
            // that removes entries by index
            let key = self.codebook.row_names[ix].clone();
            self.codebook.row_names.remove(&key);
        })
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
    /// When extending the support of a categorical column with a `value_map`,
    /// you must supply an entry in `column_metadata` for that column, and the
    /// entry must have a `value_map` that contains mapping for all valid
    /// values including those being added.
    ///
    /// # Arguments
    /// - rows: The rows of data containing the cells to insert or re-write
    /// - new_metadata: Contains the column metadata for columns to be
    ///   inserted. The columns will be inserted in the order they appear in
    ///   the metadata list. If there are columns that appear in the
    ///   column metadata that do not appear in `rows`, it will cause an error.
    /// - suppl_metadata: Contains the column metadata, indexed by column name,
    ///   for columns to be edited. For example, suppl_metadata would include
    ///   `ColMetadata` for a categorical column that was getting its support
    ///   extended.
    /// - mode: Defines how states may be modified.
    ///
    /// # Example
    ///
    /// Add a pegasus row with a few important values.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    /// use braid_data::Datum;
    /// use braid::{Row, Value, WriteMode};
    ///
    /// let mut engine = Example::Animals.engine().unwrap();
    /// let starting_rows = engine.n_rows();
    ///
    /// let rows = vec![
    ///     Row {
    ///         row_ix: "pegasus".into(),
    ///         values: vec![
    ///             Value {
    ///                 col_ix: "flys".into(),
    ///                 value: Datum::Categorical(1),
    ///             },
    ///             Value {
    ///                 col_ix: "hooves".into(),
    ///                 value: Datum::Categorical(1),
    ///             },
    ///             Value {
    ///                 col_ix: "swims".into(),
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
    ///     None,
    ///     WriteMode::unrestricted()
    /// );
    ///
    /// assert!(result.is_ok());
    /// assert_eq!(engine.n_rows(), starting_rows + 1);
    /// ```
    ///
    /// Add a column that may help us categorize a new type of animal. Note
    /// that Rows can be constructed from other simpler representations.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_data::Datum;
    /// # use braid::{Row, WriteMode};
    /// # use braid::OracleT;
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// # let starting_rows = engine.n_rows();
    /// use std::convert::TryInto;
    /// use braid_codebook::{ColMetadataList, ColMetadata, ColType};
    /// use braid_stats::prior::csd::CsdHyper;
    ///
    /// let rows: Vec<Row> = vec![
    ///     ("bat", vec![("drinks+blood", Datum::Categorical(1))]).into(),
    ///     ("beaver", vec![("drinks+blood", Datum::Categorical(0))]).into(),
    /// ];
    ///
    /// // The partial codebook is required to define the data type and
    /// // distribution of new columns
    /// let col_metadata = ColMetadataList::new(
    ///     vec![
    ///         ColMetadata {
    ///             name: "drinks+blood".into(),
    ///             coltype: ColType::Categorical {
    ///                 k: 2,
    ///                 hyper: Some(CsdHyper::default()),
    ///                 prior: None,
    ///                 value_map: None,
    ///             },
    ///             notes: None,
    ///         }
    ///     ]
    /// ).unwrap();
    /// let starting_cols = engine.n_cols();
    ///
    /// // Allow insert_data to add new columns, but not new rows, and prevent
    /// // any existing data (even missing cells) from being overwritten.
    /// let result = engine.insert_data(
    ///     rows,
    ///     Some(col_metadata),
    ///     None,
    ///     WriteMode::unrestricted(),
    /// );
    ///
    /// assert!(result.is_ok());
    /// assert_eq!(engine.n_cols(), starting_cols + 1);
    /// ```
    ///
    /// Add several new columns.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_data::Datum;
    /// # use braid::{Row, WriteMode};
    /// # use braid::OracleT;
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// # let starting_rows = engine.n_rows();
    /// use std::convert::TryInto;
    /// use braid_codebook::{ColMetadataList, ColMetadata, ColType};
    /// use braid_stats::prior::csd::CsdHyper;
    ///
    /// let rows: Vec<Row> = vec![
    ///     ("bat", vec![
    ///             ("drinks+blood", Datum::Categorical(1)),
    ///     ]).into(),
    ///     ("wolf", vec![
    ///             ("drinks+blood", Datum::Categorical(1)),
    ///             ("howls+at+the+moon", Datum::Categorical(1)),
    ///     ]).into(),
    /// ];
    ///
    /// // The partial codebook is required to define the data type and
    /// // distribution of new columns. It must contain metadata for only the
    /// // new columns.
    /// let col_metadata = ColMetadataList::new(
    ///     vec![
    ///         ColMetadata {
    ///             name: "drinks+blood".into(),
    ///             coltype: ColType::Categorical {
    ///                 k: 2,
    ///                 hyper: Some(CsdHyper::default()),
    ///                 prior: None,
    ///                 value_map: None,
    ///             },
    ///             notes: None,
    ///         },
    ///         ColMetadata {
    ///             name: "howls+at+the+moon".into(),
    ///             coltype: ColType::Categorical {
    ///                 k: 2,
    ///                 hyper: Some(CsdHyper::default()),
    ///                 prior: None,
    ///                 value_map: None,
    ///             },
    ///             notes: None,
    ///         }
    ///     ]
    /// ).unwrap();
    /// let starting_cols = engine.n_cols();
    ///
    /// // Allow insert_data to add new columns, but not new rows, and prevent
    /// // any existing data (even missing cells) from being overwritten.
    /// let result = engine.insert_data(
    ///     rows,
    ///     Some(col_metadata),
    ///     None,
    ///     WriteMode::unrestricted(),
    /// );
    ///
    /// assert!(result.is_ok());
    /// assert_eq!(engine.n_cols(), starting_cols + 2);
    /// ```
    ///
    /// We could also insert to a new category. In the animals data set all
    /// values are binary, {0, 1}. What if we decided a pig was neither fierce
    /// or docile, that it was something else, that we will capture with the
    /// value '2'?
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_data::Datum;
    /// # use braid::{Row, WriteMode};
    /// # use braid::OracleT;
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// use braid::examples::animals;
    ///
    /// // Get the value before we edit.
    /// let x_before = engine.datum(
    ///     animals::Row::Pig.into(),
    ///     animals::Column::Fierce.into()
    /// ).unwrap();
    ///
    /// // Turns out pigs are fierce.
    /// assert_eq!(x_before, Datum::Categorical(1));
    ///
    /// let rows: Vec<Row> = vec![
    ///     // Inserting a 2 into a binary column
    ///     ("pig", vec![("fierce", Datum::Categorical(2))]).into(),
    /// ];
    ///
    /// let result = engine.insert_data(
    ///     rows,
    ///     None,
    ///     None,
    ///     WriteMode::unrestricted(),
    /// );
    ///
    /// assert!(result.is_ok());
    ///
    /// // Make sure that the 2 exists in the table
    /// let x_after = engine.datum(
    ///     animals::Row::Pig.into(),
    ///     animals::Column::Fierce.into()
    /// ).unwrap();
    ///
    /// assert_eq!(x_after, Datum::Categorical(2));
    /// ```
    ///
    /// To add a category to a column with value_map
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_data::Datum;
    /// # use braid::{Row, WriteMode};
    /// # use braid::OracleT;
    /// let mut engine = Example::Satellites.engine().unwrap();
    /// use braid_codebook::{ColMetadata, ColType};
    /// use std::collections::HashMap;
    /// use maplit::{hashmap, btreemap};
    ///
    /// let suppl_metadata = {
    ///     let suppl_value_map = btreemap! {
    ///         0 => String::from("Elliptical"),
    ///         1 => String::from("GEO"),
    ///         2 => String::from("MEO"),
    ///         3 => String::from("LEO"),
    ///         4 => String::from("Lagrangian"),
    ///     };
    ///
    ///     let colmd = ColMetadata {
    ///         name: "Class_of_Orbit".into(),
    ///         notes: None,
    ///         coltype: ColType::Categorical {
    ///             k: 5,
    ///             hyper: None,
    ///             prior: None,
    ///             value_map: Some(suppl_value_map),
    ///         }
    ///     };
    ///
    ///     hashmap! {
    ///         "Class_of_Orbit".into() => colmd
    ///     }
    /// };
    ///
    /// let rows: Vec<Row> = vec![(
    ///     "Artemis (Advanced Data Relay and Technology Mission Satellite)",
    ///     vec![("Class_of_Orbit", Datum::Categorical(4))]
    /// ).into()];
    ///
    /// let result = engine.insert_data(
    ///     rows,
    ///     None,
    ///     Some(suppl_metadata),
    ///     WriteMode::unrestricted(),
    /// );
    ///
    /// assert!(result.is_ok());
    /// ```
    pub fn insert_data(
        &mut self,
        rows: Vec<Row>,
        new_metadata: Option<ColMetadataList>,
        suppl_metadata: Option<HashMap<String, ColMetadata>>,
        mode: WriteMode,
    ) -> Result<InsertDataActions, InsertDataError> {
        use super::data::standardize_rows_for_insert;
        // TODO: Lots of opportunity for optimization
        // TODO: Errors not caught
        // - user inserts missing data into new column so the column is all
        //   missing data, which wold probably break transitions
        // - user insert missing data into new row so that the row is all
        //   missing data. This might not break the transitions, but it is
        //   wasteful.

        // Convert the indices into usize if present and string/name if not
        // Error if the user has passed an usize index that is out of bounds
        let rows = standardize_rows_for_insert(rows, &self.codebook)?;

        // Figure out the tasks required to insert these data, and convert all
        // String row/col indices into usize.
        let (tasks, mut ix_rows) =
            insert_data_tasks(&rows, &new_metadata, &self)?;

        // Make sure the tasks required line up with the user-defined insert
        // mode.
        tasks.validate_insert_mode(mode)?;

        // Extend the support of categorical columns if required and allowed.
        let support_extensions =
            maybe_add_categories(&rows, &suppl_metadata, self, mode)?;

        // Add empty columns to the Engine if needed
        append_empty_columns(&tasks, new_metadata, self)?;

        // Create empty rows if needed
        if !tasks.new_rows.is_empty() {
            let n_new_rows = tasks.new_rows.len();
            self.states
                .iter_mut()
                .for_each(|state| state.extend_cols(n_new_rows));

            // NOTE: assumes the function would have already errored if row
            // names were not in the codebook
            tasks
                .new_rows
                .iter()
                .try_for_each(|row_name| {
                    self.codebook.row_names.insert(row_name.to_owned())
                })
                .expect("Somehow tried to add new row that already exists");
        }

        // Start inserting data
        ix_rows.iter_mut().try_for_each(|ixrow| {
            let row_ix = &ixrow.row_ix;
            ixrow.values.drain(..).try_for_each(|value| {
                self.insert_datum(*row_ix, value.col_ix, value.value)
            })
        })?;

        // Find all unassigned (new) rows and re-assign them
        let mut rng = &mut self.rng;
        self.states
            .iter_mut()
            .for_each(|state| state.assign_unassigned(&mut rng));

        // Remove rows from the front if we need to
        match mode.append_strategy {
            AppendStrategy::Window => {
                self.del_rows_at(0, tasks.new_rows.len());
            }
            AppendStrategy::Trench {
                max_n_rows,
                trench_ix,
            } => {
                let n_rows = self.states[0].n_rows();
                let n_remove = n_rows.saturating_sub(max_n_rows);
                self.del_rows_at(trench_ix, n_remove);
            }
            _ => (),
        }

        Ok(InsertDataActions {
            new_cols: tasks.new_cols,
            new_rows: tasks.new_rows,
            support_extensions,
        })
    }

    /// Remove data from the engine
    ///
    /// # Notes
    /// - Removing a `Datum::Missing` cell will do nothing
    /// - Removing all the cells in a row or column will completely remove that
    ///   row or column
    ///
    /// # Arguments
    /// - indices: A `Vec` of `Index`.
    ///
    /// # Example
    ///
    /// Remove a cell.
    /// ```rust
    /// # use braid::examples::Example;
    /// use braid::examples::animals::{Row, Column};
    /// use braid::{TableIndex, NameOrIndex, OracleT};
    /// use braid_data::Datum;
    ///
    /// let horse: usize = Row::Horse.into();
    /// let flys: usize = Column::Flys.into();
    ///
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// assert_eq!(engine.datum(horse, flys).unwrap(), Datum::Categorical(0));
    ///
    /// // Row and Column implement Into<TableIndex>
    /// engine.remove_data(vec![(horse, flys).into()]);
    ///
    /// assert_eq!(engine.datum(horse, flys).unwrap(), Datum::Missing);
    /// ```
    ///
    /// Remove a row and column.
    ///
    /// ```rust
    /// # use braid::examples::Example;
    /// # use braid::examples::animals::{Row, Column};
    /// # use braid::{TableIndex, NameOrIndex, OracleT};
    /// # use braid_data::Datum;
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// assert_eq!(engine.n_rows(), 50);
    /// assert_eq!(engine.n_cols(), 85);
    ///
    /// // Row and Column implement Into<TableIndex>
    /// engine.remove_data(vec![
    ///     Row::Horse.into(),
    ///     Column::Flys.into(),
    /// ]);
    ///
    /// assert_eq!(engine.n_rows(), 49);
    /// assert_eq!(engine.n_cols(), 84);
    /// ```
    ///
    /// Removing all the cells in a row, will delete the row
    ///
    /// ```rust
    /// # use braid::examples::Example;
    /// # use braid::examples::animals::{Row, Column};
    /// # use braid::{TableIndex, NameOrIndex, OracleT};
    /// # use braid_data::Datum;
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// assert_eq!(engine.n_rows(), 50);
    /// assert_eq!(engine.n_cols(), 85);
    ///
    /// // You can convert a tuple of (row_ix, col_ix) to an Index
    /// let ixs = (0..engine.n_cols())
    ///     .map(|ix| TableIndex::from((6, ix)))
    ///     .collect();
    ///
    /// engine.remove_data(ixs);
    ///
    /// assert_eq!(engine.n_rows(), 49);
    /// assert_eq!(engine.n_cols(), 85);
    /// ```
    pub fn remove_data(
        &mut self,
        mut indices: Vec<TableIndex>,
    ) -> Result<(), RemoveDataError> {
        // We use hashset because btreeset doesn't have drain. we use btreeset,
        // becuase it maintains the order of elements.
        use crate::interface::engine::data::{remove_cell, remove_col};
        use crate::{ColumnIndex, NameOrIndex, RowIndex};
        use std::collections::{BTreeSet, HashSet};

        let (rm_rows, rm_cols, rm_cells) = {
            // Get the unique indices. We could have the user provide a hash set,
            // but slices are easier to work with, so we do it here.
            let mut indices: HashSet<TableIndex> = indices.drain(..).collect();

            // TODO: return error if .to_usize_index ever returns None. that
            // means that the index was not found, so it should error rather
            // than ignore
            let mut rm_rows: BTreeSet<usize> = indices
                .iter()
                .cloned()
                .filter(|ix| ix.is_row())
                .filter_map(|ix| ix.to_usize_index(&self.codebook))
                .map(|ix| match ix {
                    TableIndex::Row(RowIndex(NameOrIndex::Index(ix))) => ix,
                    _ => panic!("Should be row index"),
                })
                .collect();

            let mut rm_cols: BTreeSet<usize> = indices
                .iter()
                .cloned()
                .filter(|ix| ix.is_column())
                .filter_map(|ix| ix.to_usize_index(&self.codebook))
                .map(|ix| match ix {
                    TableIndex::Column(ColumnIndex(NameOrIndex::Index(ix))) => {
                        ix
                    }
                    _ => panic!("Should be column index"),
                })
                .collect();

            // TODO: there is so much work happening here to figure out whether
            // we've deleted all the remaining occupied cells in a row or
            // column. It would be a lot faster if we had two counters that
            // showed for each row and column how many present data there were
            // left. Then we could increment and decrement as a part of
            // inser_data and remove_data.
            // Count the number of cells in each row and column that has been
            // removed in this operation
            let mut rm_cell_rows: HashMap<usize, i64> = HashMap::new();
            let mut rm_cell_cols: HashMap<usize, i64> = HashMap::new();

            let mut rm_cells: Vec<(usize, usize)> = indices
                .drain()
                .filter_map(|ix| ix.to_usize_index(&self.codebook))
                .filter_map(|ix| match ix {
                    TableIndex::Cell(
                        RowIndex(NameOrIndex::Index(row_ix)),
                        ColumnIndex(NameOrIndex::Index(col_ix)),
                    ) if !(rm_rows.contains(&row_ix)
                        || rm_cols.contains(&col_ix)) =>
                    {
                        rm_cell_rows
                            .entry(row_ix)
                            .and_modify(|e| *e += 1)
                            .or_insert(1);

                        rm_cell_cols
                            .entry(col_ix)
                            .and_modify(|e| *e += 1)
                            .or_insert(1);

                        Some((row_ix, col_ix))
                    }
                    _ => None,
                })
                .collect();

            use crate::interface::engine::data::check_if_removes_col;
            use crate::interface::engine::data::check_if_removes_row;
            let rows_cell_rmed =
                check_if_removes_row(&self, &rm_cols, rm_cell_rows);
            let cols_cell_rmed =
                check_if_removes_col(&self, &rm_rows, rm_cell_cols);

            rows_cell_rmed.iter().for_each(|&ix| {
                rm_rows.insert(ix);
            });
            cols_cell_rmed.iter().for_each(|&ix| {
                rm_cols.insert(ix);
            });

            let rm_cells: Vec<(usize, usize)> = rm_cells
                .drain(..)
                .filter(|(row_ix, col_ix)| {
                    !(rows_cell_rmed.contains(&row_ix)
                        || cols_cell_rmed.contains(&col_ix))
                })
                .collect();

            (rm_rows, rm_cols, rm_cells)
        };

        rm_cells
            .iter()
            .for_each(|&(row_ix, col_ix)| remove_cell(self, row_ix, col_ix));
        // Iterate through rows and cols to remove in reverse order so we don't
        // have to do more bookkeeping to account for the present index
        rm_rows
            .iter()
            .rev()
            .for_each(|&row_ix| self.del_rows_at(row_ix, 1));

        rm_cols
            .iter()
            .rev()
            .for_each(|&col_ix| remove_col(self, col_ix));

        Ok(())
    }

    /// Run the Gibbs reassignment kernel on a specific column and row withing
    /// a view. Used when the user would like to focus more updating on
    /// specific regions of the table.
    ///
    /// # Notes
    /// - The entire column will be reassigned, but only the part of row within
    ///   the view to which column `col_ix` is assigned will be updated.
    /// - Do not use a part of Geweke. This function assumes all transitions
    ///   will be run, so it cannot be guaranteed to be valid for all Geweke
    ///   configurations.
    pub fn cell_gibbs(&mut self, row_ix: usize, col_ix: usize) {
        // OracleT trait contains the is_empty() method
        use crate::OracleT as _;

        if self.is_empty() {
            return;
        }

        let mut trngs: Vec<Xoshiro256Plus> = (0..self.nstates())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng).unwrap())
            .collect();

        // rayon has a hard time doing self.states.par_iter().zip(..), so we
        // grab some mutable references explicitly
        self.states
            .par_iter_mut()
            .zip(trngs.par_iter_mut())
            .for_each(|(state, mut trng)| {
                state.reassign_col_gibbs(col_ix, true, &mut trng);
                let mut view = {
                    let view_ix = state.asgn.asgn[col_ix];
                    &mut state.views[view_ix]
                };

                view.reassign_row_gibbs(row_ix, &mut trng);

                // Make sure the view weights are correct so oracle functions
                // reflect the update correctly.
                view.weights = view.asgn.weights();
                debug_assert!(view.asgn.validate().is_valid());
            });
    }

    /// Save the Engine to a braidfile
    pub fn save_to(self, dir: &Path) -> EngineSaver {
        EngineSaver::new(self, dir)
    }

    /// Run each `State` in the `Engine` for `n_iters` iterations using the
    /// default algorithms and transitions. If the Engine is empty, `update`
    /// will immediately return.
    pub fn run(&mut self, n_iters: usize) {
        // OracleT trait contains the is_empty() method
        use crate::OracleT as _;

        if self.is_empty() {
            return;
        }

        let config = EngineUpdateConfig {
            n_iters,
            ..Default::default()
        };
        self.update(config);
    }

    /// Run each `State` in the `Engine` according to the config. If the
    /// `Engine` is empty, `update` will return immediately.
    pub fn update(&mut self, config: EngineUpdateConfig) {
        // OracleT trait contains the is_empty() method
        use crate::OracleT as _;

        if self.is_empty() {
            return;
        }

        let mut trngs: Vec<Xoshiro256Plus> = (0..self.nstates())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng).unwrap())
            .collect();

        // rayon has a hard time doing self.states.par_iter().zip(..), so we
        // grab some mutable references explicitly
        self.states
            .par_iter_mut()
            .zip(trngs.par_iter_mut())
            .for_each(|(state, mut trng)| {
                state.update(config.state_config(), &mut trng);
            });
    }

    /// Flatten the column assignment of each state so that each state has only
    /// one view
    pub fn flatten_cols(&mut self) {
        use crate::OracleT as _;

        if self.is_empty() {
            return;
        }

        let mut trngs: Vec<Xoshiro256Plus> = (0..self.nstates())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng).unwrap())
            .collect();

        self.states
            .par_iter_mut()
            .zip(trngs.par_iter_mut())
            .for_each(|(state, mut trng)| {
                state.flatten_cols(&mut trng);
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
        use braid_metadata::MetadataVersion;
        let file_config = FileConfig {
            serialized_type: self.serialized_type,
            // TODO: make sure that the metadata version aligns with what is
            // saved. I think having the most up-to-date metadata behind
            // braid_metadata::latest would be a good idea.
            version: Metadata::metadata_version(),
        };

        let dir = self.dir.as_path();
        let dir_str = dir.to_str().unwrap();

        file_utils::path_validator(dir)?;
        file_utils::save_file_config(dir, &file_config)?;

        let data = self.engine.states.get(0).unwrap().clone_data();
        file_utils::save_data(&dir, &data, &file_config)?;

        info!("Saving codebook to {}...", dir_str);
        file_utils::save_codebook(&dir, &self.engine.codebook)?;

        info!("Saving rng to {}...", dir_str);
        file_utils::save_rng(&dir, &self.engine.rng)?;

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

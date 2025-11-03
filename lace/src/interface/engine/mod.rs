mod builder;
mod data;
pub mod error;
pub mod update_handler;

pub use builder::{BuildEngineError, EngineBuilder};
pub use data::{
    AppendStrategy, InsertDataActions, InsertMode, OverwriteMode, Row,
    SupportExtension, Value, WriteMode,
};

use std::collections::HashMap;
use std::path::Path;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cc::feature::{ColModel, Feature};
use crate::cc::state::State;
use crate::codebook::{Codebook, ColMetadataList};
use crate::config::EngineUpdateConfig;
use crate::data::DataSource;
use crate::data::{Category, Datum, SummaryStatistics};
use crate::error::IndexError;
use crate::index::{ColumnIndex, RowIndex};
use crate::interface::oracle::utils::post_process_datum;
use crate::metadata::latest::Metadata;
use crate::metadata::SerializedType;
use crate::{HasData, HasStates, Oracle, TableIndex};
use data::{append_empty_columns, insert_data_tasks, maybe_add_categories};
use error::{DataParseError, InsertDataError, NewEngineError, RemoveDataError};
use polars::frame::DataFrame;

use self::update_handler::UpdateHandler;

use super::HasCodebook;

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
    fn from(oracle: Oracle) -> Self {
        Self {
            state_ids: (0..oracle.states.len()).collect(),
            states: oracle.states,
            codebook: oracle.codebook,
            rng: Xoshiro256Plus::from_os_rng(),
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
        let view_ix = state.asgn().asgn[ix];
        // XXX: Cloning the data could be very slow
        state.views[view_ix].ftrs[&ix].clone_data().summarize()
    }

    #[inline]
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum {
        let x = self.states[0].datum(row_ix, col_ix);
        post_process_datum(x, col_ix, self.codebook())
    }
}

impl HasCodebook for Engine {
    fn codebook(&self) -> &Codebook {
        &self.codebook
    }
}

#[cfg(feature = "formats")]
fn col_models_from_data_src<R: rand::Rng>(
    codebook: Codebook,
    data_source: DataSource,
    rng: &mut R,
) -> Result<(Codebook, Vec<ColModel>), DataParseError> {
    use crate::codebook::formats;
    let df = match data_source {
        DataSource::Csv(path) => formats::read_csv(path).unwrap(),
        DataSource::Ipc(path) => formats::read_ipc(path).unwrap(),
        DataSource::Json(path) => formats::read_json(path).unwrap(),
        DataSource::Parquet(path) => formats::read_parquet(path).unwrap(),
        DataSource::Polars(df) => df,
        DataSource::Empty => DataFrame::empty(),
    };
    crate::data::df_to_col_models(codebook, df, rng)
}

#[cfg(not(feature = "formats"))]
fn col_models_from_data_src<R: rand::Rng>(
    codebook: Codebook,
    data_source: DataSource,
    rng: &mut R,
) -> Result<(Codebook, Vec<ColModel>), DataParseError> {
    let df = match data_source {
        DataSource::Polars(df) => df,
        DataSource::Empty => DataFrame::empty(),
    };
    crate::data::df_to_col_models(codebook, df, rng)
}

fn emit_prior_process<R: rand::Rng>(
    prior_process: crate::codebook::PriorProcess,
    rng: &mut R,
) -> crate::stats::prior_process::Process {
    use crate::stats::prior_process::{Dirichlet, PitmanYor, Process};
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

/// Maintains and samples states
impl Engine {
    /// Create a new engine
    ///
    /// # Arguments
    /// - n_states: number of states
    /// - id_offset: the state IDs will start at `id_offset`. This is useful
    ///   for when you run multiple engines on multiple machines and want to
    ///   easily combine the states in a single `Oracle` after the runs
    /// - data_source: struct defining the type or data and path
    /// - id_offset: the state IDs will be `0+id_offset, ..., n_states +
    ///   id_offset`. If offset is helpful when you want to run a single model
    ///   on multiple machines and merge the states into the same metadata
    ///   folder.
    /// - rng: Random number generator
    pub fn new(
        n_states: usize,
        codebook: Codebook,
        data_source: DataSource,
        id_offset: usize,
        mut rng: Xoshiro256Plus,
    ) -> Result<Self, NewEngineError> {
        if n_states == 0 {
            return Err(NewEngineError::ZeroStatesRequested);
        }

        let (codebook, col_models) =
            col_models_from_data_src(codebook, data_source, &mut rng)
                .map_err(NewEngineError::DataParseError)?;

        let state_prior_process = emit_prior_process(
            codebook.state_prior_process.clone().unwrap_or_default(),
            &mut rng,
        );

        let view_prior_process = emit_prior_process(
            codebook.view_prior_process.clone().unwrap_or_default(),
            &mut rng,
        );

        let states: Vec<State> = (0..n_states)
            .map(|_| {
                let features = col_models.clone();
                State::from_prior(
                    features,
                    state_prior_process.clone(),
                    view_prior_process.clone(),
                    &mut rng,
                )
            })
            .collect();

        let state_ids = (id_offset..n_states + id_offset).collect();

        Ok(Self {
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

    ///  Load a lacefile into an `Engine`.
    pub fn load<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, crate::metadata::Error> {
        let metadata = crate::metadata::load_metadata(path)?;
        metadata
            .try_into()
            .map_err(|err| crate::metadata::Error::Other(format!("{err}")))
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

        let mut rng = self.rng.clone();

        self.states
            .iter_mut()
            .for_each(|state| state.del_rows_at(ix, n, &mut rng));

        (0..n).for_each(|_| {
            // TODO: get rid of this clone by adding a method to RowNameList
            // that removes entries by index
            let key = self.codebook.row_names[ix].clone();
            self.codebook.row_names.remove(&key);
        })
    }

    /// Delete the column at `col_ix`
    ///
    /// # Example
    ///
    /// ```
    /// use lace::examples::Example;
    /// use lace::OracleT;
    ///
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// let shape = engine.shape();
    /// assert_eq!(shape, (50, 85, 16));
    ///
    /// // String index
    /// engine.del_column("swims");
    /// assert_eq!(engine.shape(), (50, 84, 16));
    ///
    /// // Integer index
    /// engine.del_column(3);
    /// assert_eq!(engine.shape(), (50, 83, 16));
    /// ```
    ///
    /// Deleting a column that does not exist returns and `IndexError`
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// let result = engine.del_column("likes_milk_in_coffee");
    /// assert!(result.is_err());
    /// ```
    pub fn del_column<Ix: ColumnIndex>(
        &mut self,
        col_ix: Ix,
    ) -> Result<(), IndexError> {
        col_ix
            .col_ix(&self.codebook)
            .map(|ix| data::remove_col(self, ix))
    }

    /// Insert a datum at the provided index
    fn insert_datum(
        &mut self,
        row_ix: usize,
        col_ix: usize,
        datum: Datum,
    ) -> Result<(), InsertDataError> {
        // Handle the case when we have to convert the datum to a value that can
        // be understood by the state. States currently only hold categorical
        // data as u32, so we have to convert from String or Bool.
        let datum = if let Datum::Categorical(ref cat) = datum {
            let ix: usize = self
                .codebook
                .value_map(col_ix)
                .ok_or_else(|| {
                    InsertDataError::ColumnIndex(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: self.n_cols(),
                            col_ix,
                        },
                    )
                })
                .and_then(|vm| {
                    vm.ix(cat).ok_or_else(|| {
                        dbg!(&vm);
                        InsertDataError::CategoryNotInValueMap(cat.clone())
                    })
                })?;
            Datum::Categorical(Category::UInt(ix as u32))
        } else {
            datum
        };

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
    /// - mode: Defines how states may be modified.
    ///
    /// # Example
    ///
    /// Add a pegasus row with a few important values.
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::{OracleT, HasStates};
    /// use lace::data::Datum;
    /// use lace::{Row, Value, WriteMode};
    ///
    /// let mut engine = Example::Animals.engine().unwrap();
    /// let starting_rows = engine.n_rows();
    ///
    /// let rows = vec![
    ///     Row::<&str, &str> {
    ///         row_ix: "pegasus".into(),
    ///         values: vec![
    ///             Value {
    ///                 col_ix: "flys".into(),
    ///                 value: Datum::Categorical(1_u32.into()),
    ///             },
    ///             Value {
    ///                 col_ix: "hooves".into(),
    ///                 value: Datum::Categorical(1_u32.into()),
    ///             },
    ///             Value {
    ///                 col_ix: "swims".into(),
    ///                 value: Datum::Categorical(0_u32.into()),
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
    /// # use lace::examples::Example;
    /// # use lace::data::Datum;
    /// # use lace::{Row, WriteMode};
    /// # use lace::{OracleT, HasStates};
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// # let starting_rows = engine.n_rows();
    /// use lace::codebook::{ColMetadataList, ColMetadata, ColType, ValueMap};
    /// use lace::stats::prior::csd::CsdHyper;
    ///
    /// let rows: Vec<Row<&str, &str>> = vec![
    ///     ("bat", vec![("drinks+blood", Datum::Categorical(1_u32.into()))]).into(),
    ///     ("beaver", vec![("drinks+blood", Datum::Categorical(0_u32.into()))]).into(),
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
    ///                 value_map: ValueMap::UInt(2),
    ///             },
    ///             notes: None,
    ///             missing_not_at_random: false,
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
    /// # use lace::examples::Example;
    /// # use lace::data::Datum;
    /// # use lace::{Row, WriteMode};
    /// # use lace::{OracleT, HasStates};
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// # let starting_rows = engine.n_rows();
    /// use lace::codebook::{ColMetadataList, ColMetadata, ColType, ValueMap};
    /// use lace::stats::prior::csd::CsdHyper;
    ///
    /// let rows: Vec<Row<&str, &str>> = vec![
    ///     ("bat", vec![
    ///             ("drinks+blood", Datum::Categorical(1_u32.into())),
    ///     ]).into(),
    ///     ("wolf", vec![
    ///             ("drinks+blood", Datum::Categorical(1_u32.into())),
    ///             ("howls+at+the+moon", Datum::Categorical(1_u32.into())),
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
    ///                 value_map: ValueMap::UInt(2),
    ///             },
    ///             notes: None,
    ///             missing_not_at_random: false,
    ///         },
    ///         ColMetadata {
    ///             name: "howls+at+the+moon".into(),
    ///             coltype: ColType::Categorical {
    ///                 k: 2,
    ///                 hyper: Some(CsdHyper::default()),
    ///                 prior: None,
    ///                 value_map: ValueMap::UInt(2),
    ///             },
    ///             notes: None,
    ///             missing_not_at_random: false,
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
    /// # use lace::examples::Example;
    /// # use lace::data::Datum;
    /// # use lace::{Row, WriteMode};
    /// # use lace::OracleT;
    /// # let mut engine = Example::Animals.engine().unwrap();
    /// use lace::examples::animals;
    ///
    /// // Get the value before we edit.
    /// let x_before = engine.datum("pig", "fierce").unwrap();
    ///
    /// // Turns out pigs are fierce.
    /// assert_eq!(x_before, Datum::Categorical(1_u32.into()));
    ///
    /// let rows: Vec<Row<&str, &str>> = vec![
    ///     // Inserting a 2 into a binary column
    ///     ("pig", vec![("fierce", Datum::Categorical(2_u32.into()))]).into(),
    /// ];
    ///
    /// let result = engine.insert_data(
    ///     rows,
    ///     None,
    ///     WriteMode::unrestricted(),
    /// );
    ///
    /// assert!(result.is_ok());
    ///
    /// // Make sure that the 2 exists in the table
    /// let x_after = engine.datum("pig", "fierce").unwrap();
    ///
    /// assert_eq!(x_after, Datum::Categorical(2_u32.into()));
    /// ```
    ///
    /// To add a category to a column with value_map
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::data::Datum;
    /// # use lace::{Row, WriteMode};
    /// # use lace::OracleT;
    /// let mut engine = Example::Satellites.engine().unwrap();
    /// use lace::codebook::{ColMetadata, ColType, ValueMap};
    /// use std::collections::HashMap;
    ///
    /// let rows: Vec<Row<&str, &str>> = vec![(
    ///     "Artemis (Advanced Data Relay and Technology Mission Satellite)",
    ///     vec![("Class_of_Orbit", Datum::Categorical("MEO".into()))]
    /// ).into()];
    ///
    /// let result = engine.insert_data(
    ///     rows,
    ///     None,
    ///     WriteMode::unrestricted(),
    /// );
    /// assert!(result.is_ok());
    /// ```
    pub fn insert_data<R: RowIndex, C: ColumnIndex>(
        &mut self,
        rows: Vec<Row<R, C>>,
        new_metadata: Option<ColMetadataList>,
        mode: WriteMode,
    ) -> Result<InsertDataActions, InsertDataError> {
        // use data::standardize_rows_for_insert;
        // TODO: Lots of opportunity for optimization
        // TODO: Errors not caught
        // - user inserts missing data into new column so the column is all
        //   missing data, which wold probably break transitions
        // - user insert missing data into new row so that the row is all
        //   missing data. This might not break the transitions, but it is
        //   wasteful.

        // Convert the indices into usize if present and string/name if not
        // Error if the user has passed an usize index that is out of bounds
        // let rows = standardize_rows_for_insert(rows, &self.codebook)?;

        // Figure out the tasks required to insert these data, and convert all
        // String row/col indices into usize.
        let (tasks, mut ix_rows) =
            insert_data_tasks(&rows, &new_metadata, self)?;

        // Make sure the tasks required line up with the user-defined insert
        // mode.
        tasks.validate_insert_mode(mode)?;

        // Extend the support of categorical columns if required and allowed.
        let support_extensions = maybe_add_categories(&rows, self, mode)?;

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
                    self.codebook.row_names.insert(row_name.clone())
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
    /// # use lace::examples::Example;
    /// use lace::{TableIndex, OracleT};
    /// use lace::data::Datum;
    ///
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// assert_eq!(
    ///     engine.datum("horse", "flys").unwrap(),
    ///     Datum::Categorical(0_u32.into()),
    /// );
    ///
    /// // Row and Column implement Into<TableIndex>
    /// engine.remove_data(vec![("horse", "flys").into()]);
    ///
    /// assert_eq!(engine.datum("horse", "flys").unwrap(), Datum::Missing);
    /// ```
    ///
    /// Remove a row and column.
    ///
    /// ```rust
    /// # use lace::examples::Example;
    /// # use lace::{TableIndex, OracleT, HasStates};
    /// # use lace::data::Datum;
    /// let mut engine = Example::Animals.engine().unwrap();
    ///
    /// assert_eq!(engine.n_rows(), 50);
    /// assert_eq!(engine.n_cols(), 85);
    ///
    /// engine.remove_data(vec![
    ///     TableIndex::Row("horse"),
    ///     TableIndex::Column("flys"),
    /// ]);
    ///
    /// assert_eq!(engine.n_rows(), 49);
    /// assert_eq!(engine.n_cols(), 84);
    /// ```
    ///
    /// Removing all the cells in a row, will delete the row
    ///
    /// ```rust
    /// # use lace::examples::Example;
    /// # use lace::{TableIndex, OracleT, HasStates};
    /// # use lace::data::Datum;
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
    pub fn remove_data<R: RowIndex, C: ColumnIndex>(
        &mut self,
        mut indices: Vec<TableIndex<R, C>>,
    ) -> Result<(), RemoveDataError> {
        // We use hashset because btreeset doesn't have drain. we use btreeset,
        // becuase it maintains the order of elements.
        use crate::interface::engine::data::{remove_cell, remove_col};
        use std::collections::{BTreeSet, HashSet};

        let codebook = self.codebook();

        let (rm_rows, rm_cols, rm_cells) = {
            // Get the unique indices. We could have the user provide a hash set,
            // but slices are easier to work with, so we do it here.
            let mut indices: HashSet<TableIndex<usize, usize>> = indices
                .drain(..)
                .map(|ix| ix.into_usize_index(codebook))
                .collect::<Result<_, _>>()?;

            // TODO: return error if .to_usize_index ever returns None. that
            // means that the index was not found, so it should error rather
            // than ignore
            let mut rm_rows: BTreeSet<usize> = indices
                .iter()
                .filter(|ix| ix.is_row())
                .map(|ix| match ix {
                    TableIndex::Row(ix) => *ix,
                    _ => panic!("Should be row index"),
                })
                .collect();

            let mut rm_cols: BTreeSet<usize> = indices
                .iter()
                .filter(|ix| ix.is_column())
                .map(|ix| match ix {
                    TableIndex::Column(ix) => *ix,
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
                .filter_map(|ix| match ix {
                    TableIndex::Cell(row_ix, col_ix)
                        if !(rm_rows.contains(&row_ix)
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
                check_if_removes_row(self, &rm_cols, rm_cell_rows);
            let cols_cell_rmed =
                check_if_removes_col(self, &rm_rows, rm_cell_cols);

            rows_cell_rmed.iter().for_each(|&ix| {
                rm_rows.insert(ix);
            });
            cols_cell_rmed.iter().for_each(|&ix| {
                rm_cols.insert(ix);
            });

            let rm_cells: Vec<(usize, usize)> = rm_cells
                .drain(..)
                .filter(|(row_ix, col_ix)| {
                    !(rows_cell_rmed.contains(row_ix)
                        || cols_cell_rmed.contains(col_ix))
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

    /// Run the Gibbs reassignment kernel on a specific column and row within
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

        let mut trngs: Vec<Xoshiro256Plus> = (0..self.n_states())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng))
            .collect();

        // rayon has a hard time doing self.states.par_iter().zip(..), so we
        // grab some mutable references explicitly
        self.states
            .par_iter_mut()
            .zip(trngs.par_iter_mut())
            .for_each(|(state, mut trng)| {
                state.reassign_col_gibbs(col_ix, true, &mut trng);
                let view = {
                    let view_ix = state.asgn().asgn[col_ix];
                    &mut state.views[view_ix]
                };

                view.reassign_row_gibbs(row_ix, &mut trng);

                // Make sure the view weights are correct so oracle functions
                // reflect the update correctly.
                view.weights = view.prior_process.weight_vec(false);
                debug_assert!(view.asgn().validate().is_valid());
            });
    }

    /// Save the Engine to a lacefile
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        ser_type: SerializedType,
    ) -> Result<(), crate::metadata::Error> {
        let metadata: Metadata = self.into();
        crate::metadata::save_metadata(&metadata, path, ser_type)?;
        Ok(())
    }

    /// Run each `State` in the `Engine` for `n_iters` iterations using the
    /// default algorithms and transitions. If the Engine is empty, `update`
    /// will immediately return.
    pub fn run(
        &mut self,
        n_iters: usize,
    ) -> Result<(), crate::metadata::Error> {
        // OracleT trait contains the is_empty() method
        use crate::OracleT as _;

        if self.is_empty() {
            return Ok(());
        }

        let config = EngineUpdateConfig::new()
            .default_transitions()
            .n_iters(n_iters);

        self.update(config, ())
    }

    /// Run each `State` in the `Engine` according to the config. If the
    /// `Engine` is empty, `update` will return immediately.
    pub fn update<U>(
        &mut self,
        config: EngineUpdateConfig,
        mut update_handler: U,
    ) -> Result<(), crate::metadata::Error>
    where
        U: UpdateHandler,
    {
        // OracleT trait contains the is_empty() method
        use crate::OracleT as _;

        assert!(!config.transitions.is_empty());

        if self.is_empty() {
            return Ok(());
        }

        // Initialize update_handler
        update_handler.global_init(&config, &self.states);

        // Save up frontif the the use has provided a save config. If the user
        // has also provided a checkpoint arg, we use this initial save to save
        // the data, rng state, config, etc.
        if let Some(config) = config.clone().save_config {
            self.save(config.path, config.ser_type)?;
        }

        let mut trngs: Vec<Xoshiro256Plus> = (0..self.n_states())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng))
            .collect();

        let state_config = config.state_config();

        let checkpoint_iters = config.checkpoint.unwrap_or(config.n_iters);

        let n_checkpoints = if config.n_iters % checkpoint_iters == 0 {
            config.n_iters / checkpoint_iters
        } else {
            config.n_iters / checkpoint_iters + 1
        };

        let mut update_handlers: Vec<U> = (0..self.n_states())
            .map(|_| update_handler.clone())
            .collect();

        for i in 0..n_checkpoints {
            self.states = self
                .states
                .par_drain(..)
                .zip(trngs.par_iter_mut())
                .zip(update_handlers.par_iter_mut())
                .zip(self.state_ids.par_iter())
                .map(|(((state, mut trng), handler),  &state_ix)| {
                    // how many iters to run this checkpoint
                    let total_iters = i * checkpoint_iters;
                    let n_iters =
                        if total_iters + checkpoint_iters > config.n_iters {
                            config.n_iters - total_iters
                        } else {
                            checkpoint_iters
                        };
                    handler.new_state_init(state_ix, &state);

                    (0..n_iters)
                        .try_fold(state, |mut state, iter| {
                            // Stop updating if the desired itertion has occured
                            // or an external condition has been met.
                            if state_config.check_over_iters(iter) || handler.stop_engine() || handler.stop_state(state_ix) {
                                Ok(state)
                            } else {
                                // otherwise, step
                                state.step(&config.transitions, &mut trng);
                                state.push_diagnostics();

                                // Update the update handlers.
                                handler.state_updated(state_ix, &state);
                                Ok(state)
                            }
                        })
                        .and_then(|mut state| {
                            handler.state_complete(state_ix, &state);

                            // convert state to dataless, save, and convert back
                            if let Some(config) = config.clone().save_config {
                                use crate::metadata::latest::{
                                    DatalessStateAndDiagnostics, EmptyState,
                                };

                                let (data, dataless_state) = {
                                    let data = state.take_data();
                                    let dataless_state: DatalessStateAndDiagnostics =
                                        state.into();
                                    (data, dataless_state)
                                };

                                crate::metadata::save_state(
                                    &config.path,
                                    &dataless_state,
                                    state_ix,
                                    config.ser_type,
                                )
                                .map(|_| {
                                    let empty_state: EmptyState =
                                        dataless_state.into();
                                    let mut inner = empty_state.0;
                                    inner.repop_data(data);
                                    inner
                                })
                            } else {
                                Ok(state)
                            }
                        })
                })
                .collect::<Result<Vec<State>, _>>()?;
        }
        std::mem::drop(update_handlers);
        update_handler.finalize();

        Ok(())
    }

    /// Flatten the column assignment of each state so that each state has only
    /// one view
    pub fn flatten_cols(&mut self) {
        use crate::OracleT as _;

        if self.is_empty() {
            return;
        }

        let mut trngs: Vec<Xoshiro256Plus> = (0..self.n_states())
            .map(|_| Xoshiro256Plus::from_rng(&mut self.rng))
            .collect();

        self.states
            .par_iter_mut()
            .zip(trngs.par_iter_mut())
            .for_each(|(state, mut trng)| {
                state.flatten_cols(&mut trng);
            });
    }

    /// Returns the number of states
    pub fn n_states(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        path::PathBuf,
        sync::{Arc, RwLock},
        time::Duration,
    };

    use crate::update_handler::StateTimeout;

    use super::*;

    fn animals_csv() -> DataSource {
        let df = crate::codebook::data::read_csv(PathBuf::from(
            "resources/datasets/animals/data.csv",
        ))
        .unwrap();
        DataSource::Polars(df)
    }

    #[test]
    fn all_update_handler_methods_called() {
        #[derive(Clone)]
        struct TestingHandler(Arc<RwLock<HashSet<String>>>);

        impl UpdateHandler for TestingHandler {
            fn global_init(
                &mut self,
                _config: &EngineUpdateConfig,
                _states: &[State],
            ) {
                self.0.write().unwrap().insert("global_init".to_string());
            }

            fn new_state_init(&mut self, _state_id: usize, _state: &State) {
                self.0.write().unwrap().insert("new_state_init".to_string());
            }

            fn state_updated(&mut self, _state_id: usize, _state: &State) {
                self.0.write().unwrap().insert("state_updated".to_string());
            }

            fn state_complete(&mut self, _state_id: usize, _state: &State) {
                self.0.write().unwrap().insert("state_complete".to_string());
            }

            fn stop_engine(&self) -> bool {
                self.0.write().unwrap().insert("stop_engine".to_string());
                false
            }

            fn stop_state(&self, _state_id: usize) -> bool {
                self.0.write().unwrap().insert("stop_state".to_string());
                false
            }

            fn finalize(&mut self) {
                self.0.write().unwrap().insert("finalize".to_string());
            }
        }

        let mut engine = EngineBuilder::new(animals_csv()).build().unwrap();

        let called_methods = Arc::new(RwLock::new(HashSet::new()));
        let update_handler = TestingHandler(called_methods.clone());

        let config = EngineUpdateConfig::new().default_transitions().n_iters(1);

        engine
            .update(config, update_handler)
            .expect("update() processed correctly");

        let expected_methods_called: HashSet<String> = vec![
            "global_init",
            "new_state_init",
            "state_updated",
            "state_complete",
            "stop_engine",
            "stop_state",
            "finalize",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(
            *called_methods.read().unwrap(),
            expected_methods_called,
            "All expected methods were called"
        );
    }

    // This just tests that the StateTimeout handler will not impede normal processing.
    // It does not test that the StateTimeout successfully ends states that have gone over the duration
    #[test]
    fn state_timeout_update_handler() {
        let mut engine = EngineBuilder::new(animals_csv()).build().unwrap();

        let config = EngineUpdateConfig::new().default_transitions().n_iters(1);

        let update_handler = StateTimeout::new(Duration::from_secs(3600));

        engine.update(config, update_handler).expect(
            "update() processed with the StateTimeout UpdateHandler correctly",
        );
    }
}

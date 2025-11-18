use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryInto;

use indexmap::IndexSet;
use rv::data::CategoricalSuffStat;
use rv::dist::Categorical;
use rv::dist::SymmetricDirichlet;
use serde::Deserialize;
use serde::Serialize;

use super::error::InsertDataError;
use crate::cc::feature::ColModel;
use crate::cc::feature::Column;
use crate::cc::feature::FType;
use crate::codebook::Codebook;
use crate::codebook::ColMetadataList;
use crate::codebook::ColType;
use crate::codebook::ValueMap;
use crate::codebook::ValueMapExtension;
use crate::codebook::ValueMapExtensionError;
use crate::data::Category;
use crate::data::Datum;
use crate::data::SparseContainer;
use crate::interface::HasCodebook;
use crate::ColumnIndex;
use crate::Engine;
use crate::HasStates;
use crate::OracleT;
use crate::RowIndex;

/// Defines which data may be overwritten
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverwriteMode {
    /// Overwrite anything
    Allow,
    /// Do not overwrite any existing cells. Only allow data in new rows or
    /// columns.
    Deny,
    /// Same as deny, but also allow existing cells that are empty to be
    /// overwritten.
    MissingOnly,
}

/// Defines insert data behavior -- where data may be inserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertMode {
    /// Can add new rows or column
    Unrestricted,
    /// Cannot add new rows, but can add new columns
    DenyNewRows,
    /// Cannot add new columns, but can add new rows
    DenyNewColumns,
    /// No adding new rows or columns
    DenyNewRowsAndColumns,
}

/// Defines the behavior of the data table when new rows are appended
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum AppendStrategy {
    /// New rows will be appended and the rest of the table will be unchanged
    #[default]
    None,
    /// If `n` rows are added, the top `n` rows will be removed
    Window,
    /// For each row added that exceeds `max_n_rows`, the row at `tench_ix` will
    /// be removed.
    Trench {
        /// The max number of rows allowed
        max_n_rows: usize,
        /// The index to remove data from
        trench_ix: usize,
    },
}

/// Defines how/where data may be inserted, which day may and may not be
/// overwritten, and whether data may extend the domain
///
/// # Example
///
/// Default `WriteMode` only allows appending supported values to new rows or
/// columns
/// ```
/// use lace::{WriteMode, InsertMode, OverwriteMode, AppendStrategy};
/// let mode_new = WriteMode::new();
/// let mode_def = WriteMode::default();
///
/// assert_eq!(
///     mode_new,
///     WriteMode {
///         insert: InsertMode::Unrestricted,
///         overwrite: OverwriteMode::Deny,
///         allow_extend_support: false,
///         append_strategy: AppendStrategy::None,
///     }
/// );
/// assert_eq!(mode_def, mode_new);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WriteMode {
    /// Determines whether new rows or columns can be appended or if data may
    /// be entered into existing cells.
    pub insert: InsertMode,
    /// Determines if existing cells may or may not be overwritten or whether
    /// only missing cells may be overwritten.
    pub overwrite: OverwriteMode,
    /// If `true`, allow column support to be extended to accommodate new data
    /// that fall outside the range. For example, a binary column extends to
    /// ternary after the user inserts `Datum::Categorical(2)`.
    #[serde(default)]
    pub allow_extend_support: bool,
    /// The behavior of the table when new rows are appended
    #[serde(default)]
    pub append_strategy: AppendStrategy,
}

impl WriteMode {
    /// Allows new data to be appended only to new rows/columns. No overwriting
    /// and no support extension.
    #[inline]
    pub fn new() -> Self {
        Self {
            insert: InsertMode::Unrestricted,
            overwrite: OverwriteMode::Deny,
            allow_extend_support: false,
            append_strategy: AppendStrategy::None,
        }
    }

    #[inline]
    pub fn unrestricted() -> Self {
        Self {
            insert: InsertMode::Unrestricted,
            overwrite: OverwriteMode::Allow,
            allow_extend_support: true,
            append_strategy: AppendStrategy::None,
        }
    }
}

impl Default for WriteMode {
    fn default() -> Self {
        Self::new()
    }
}

/// A datum for insertion into a certain column
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Value<C: ColumnIndex> {
    /// Name of the column
    pub col_ix: C,
    /// The value of the cell
    pub value: Datum,
}

impl<C: ColumnIndex> From<(C, Datum)> for Value<C> {
    fn from(value: (C, Datum)) -> Self {
        Self {
            col_ix: value.0,
            value: value.1,
        }
    }
}

/// A list of data for insertion into a certain row
///
/// # Example
///
/// ```
/// # use lace::Row;
/// use lace::Value;
/// use lace::data::Datum;
///
/// let row = Row::<&str, &str> {
///     row_ix: "vampire",
///     values: vec![
///         Value {
///             col_ix: "sucks_blood",
///             value: Datum::Categorical(1_u32.into()),
///         },
///         Value {
///             col_ix: "drinks_wine",
///             value: Datum::Categorical(0_u32.into()),
///         },
///     ],
/// };
///
/// assert_eq!(row.len(), 2);
/// ```
///
/// There are converters for convenience.
///
/// ```
/// # use lace::Row;
/// # use lace::data::Datum;
/// let row: Row<&str, &str>  = (
///     "vampire",
///     vec![
///         ("sucks_blood", Datum::Categorical(1_u32.into())),
///         ("drinks_wine", Datum::Categorical(0_u32.into())),
///     ]
/// ).into();
///
/// assert_eq!(row.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row<R: RowIndex, C: ColumnIndex> {
    /// The name of the row
    pub row_ix: R,
    /// The cells and values to fill in
    pub values: Vec<Value<C>>,
}

impl<R, C> From<(R, Vec<(C, Datum)>)> for Row<R, C>
where
    R: RowIndex,
    C: ColumnIndex,
{
    fn from(mut row: (R, Vec<(C, Datum)>)) -> Self {
        Self {
            row_ix: row.0,
            values: row.1.drain(..).map(Value::from).collect(),
        }
    }
}

impl<R: RowIndex, C: ColumnIndex> From<(R, Vec<Value<C>>)> for Row<R, C> {
    fn from(row: (R, Vec<Value<C>>)) -> Self {
        Self {
            row_ix: row.0,
            values: row.1,
        }
    }
}

impl<R: RowIndex, C: ColumnIndex> Row<R, C> {
    /// The number of values in the Row
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return true if there are no values in the Row
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// Because lace uses integer indices for rows and columns
#[derive(Debug, PartialEq)]
pub(crate) struct IndexValue {
    pub col_ix: usize,
    pub value: Datum,
}

#[derive(Debug, PartialEq)]
pub(crate) struct IndexRow {
    pub row_ix: usize,
    pub values: Vec<IndexValue>,
}

/// Describes the support extension action taken
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SupportExtension {
    Categorical {
        /// The index of the column
        col_ix: usize,
        /// The name of the column
        col_name: String,
        /// New mapped values
        value_map_extension: ValueMapExtension,
    },
}

/// Describes table-extending actions taken when inserting data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsertDataActions {
    // the types of the members match the types in InsertDataTasks
    pub(crate) new_rows: IndexSet<String>,
    pub(crate) new_cols: HashSet<String>,
    pub(crate) support_extensions: Vec<SupportExtension>,
}

impl Default for InsertDataActions {
    fn default() -> Self {
        Self::new()
    }
}

impl InsertDataActions {
    pub fn new() -> Self {
        Self {
            new_rows: IndexSet::new(),
            new_cols: HashSet::new(),
            support_extensions: Vec::new(),
        }
    }

    /// If any new rows were appended, returns their names and order
    pub fn new_rows(&self) -> Option<&IndexSet<String>> {
        if self.new_rows.is_empty() {
            None
        } else {
            Some(&self.new_rows)
        }
    }

    /// If any new columns were appended, returns their names
    pub fn new_cols(&self) -> Option<&HashSet<String>> {
        if self.new_cols.is_empty() {
            None
        } else {
            Some(&self.new_cols)
        }
    }

    // The any columns had their supports extended, returns the support
    // actions taken
    pub fn support_extensions(&self) -> Option<&Vec<SupportExtension>> {
        if self.support_extensions.is_empty() {
            None
        } else {
            Some(&self.support_extensions)
        }
    }
}

/// A summary of the tasks required to insert certain data into an `Engine`
#[derive(Debug)]
pub(crate) struct InsertDataTasks {
    /// The names of new rows to be created. The order of the items is the
    /// order in the which the rows are inserted.
    pub new_rows: IndexSet<String>,
    /// The names of new columns to be created
    pub new_cols: HashSet<String>,
    /// True if the operation would insert a value into an empty cell in the
    /// existing table
    pub overwrite_missing: bool,
    /// True if the operation would overwrite an existing (non-missing) value
    pub overwrite_present: bool,
}

impl InsertDataTasks {
    fn new() -> Self {
        Self {
            new_rows: IndexSet::new(),
            new_cols: HashSet::new(),
            overwrite_missing: false,
            overwrite_present: false,
        }
    }

    pub(crate) fn validate_insert_mode(
        &self,
        mode: WriteMode,
    ) -> Result<(), InsertDataError> {
        match mode.overwrite {
            OverwriteMode::Deny => {
                if self.overwrite_present || self.overwrite_missing {
                    Err(InsertDataError::ModeForbidsOverwrite)
                } else {
                    Ok(())
                }
            }
            OverwriteMode::MissingOnly => {
                if self.overwrite_present {
                    Err(InsertDataError::ModeForbidsOverwrite)
                } else {
                    Ok(())
                }
            }
            OverwriteMode::Allow => Ok(()),
        }
        .and_then(|_| match mode.insert {
            InsertMode::DenyNewRows => {
                if !self.new_rows.is_empty() {
                    Err(InsertDataError::ModeForbidsNewRows)
                } else {
                    Ok(())
                }
            }
            InsertMode::DenyNewColumns => {
                if !self.new_cols.is_empty() {
                    Err(InsertDataError::ModeForbidsNewColumns)
                } else {
                    Ok(())
                }
            }
            InsertMode::DenyNewRowsAndColumns => {
                if !(self.new_rows.is_empty() && self.new_cols.is_empty()) {
                    Err(InsertDataError::ModeForbidsNewRowsOrColumns)
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        })
    }
}

#[inline]
fn ix_lookup_from_codebook(
    col_metadata: &Option<ColMetadataList>,
) -> Option<HashMap<&str, usize>> {
    col_metadata.as_ref().map(|colmds| {
        colmds
            .iter()
            .enumerate()
            .map(|(ix, md)| (md.name.as_str(), ix))
            .collect()
    })
}

#[inline]
fn col_ix_from_lookup(
    col: &str,
    lookup: &Option<HashMap<&str, usize>>,
) -> Result<usize, InsertDataError> {
    match lookup {
        Some(lkp) => lkp
            .get(col)
            .ok_or_else(|| {
                InsertDataError::NewColumnNotInColumnMetadata(col.to_owned())
            })
            .copied(),
        None => Err(InsertDataError::NewColumnNotInColumnMetadata(
            String::from(col),
        )),
    }
}

/// Determine whether we need to add new columns to the Engine and then add
/// them.
pub(crate) fn append_empty_columns(
    tasks: &InsertDataTasks,
    col_metadata: Option<ColMetadataList>,
    engine: &mut Engine,
) -> Result<(), InsertDataError> {
    match col_metadata {
        // There is partial codebook and there are new columns to add
        Some(colmds) if !tasks.new_cols.is_empty() => {
            // make sure that each of the new columns to be added is listed in
            // the column metadata
            tasks.new_cols.iter().try_for_each(|col| {
                if colmds.contains_key(col) {
                    Ok(())
                } else {
                    Err(InsertDataError::NewColumnNotInColumnMetadata(
                        col.clone(),
                    ))
                }
            })?;

            if colmds.len() != tasks.new_cols.len() {
                // There are more columns in the partial codebook than are
                // in the inserted data.
                Err(InsertDataError::WrongNumberOfColumnMetadataEntries {
                    ncolmd: colmds.len(),
                    nnew: tasks.new_cols.len(),
                })
            } else {
                // create blank (data-less) columns and insert them into
                // the States
                let shape = (engine.n_rows(), engine.n_cols());
                create_new_columns(&colmds, shape, &mut engine.rng).map(
                    |col_models| {
                        // Inserts blank columns into random existing views.
                        // It is assumed that another reassignment transition
                        // will be run after the data are inserted.
                        let mut rng = &mut engine.rng;
                        engine.states.iter_mut().for_each(|state| {
                            state.append_blank_features(
                                col_models.clone(),
                                &mut rng,
                            );
                        });

                        // Combine the codebooks
                        // NOTE: if a panic happens here its our fault.
                        // TODO: only append the ones that are new
                        engine.codebook.append_col_metadata(colmds).unwrap();
                    },
                )
            }
        }
        // There are new columns, but no partial codebook
        None if !tasks.new_cols.is_empty() => {
            Err(InsertDataError::WrongNumberOfColumnMetadataEntries {
                ncolmd: 0,
                nnew: tasks.new_cols.len(),
            })
        }
        // Can ignore other cases (no new columns)
        _ => Ok(()),
    }
}

fn validate_new_col_ftype(
    new_metadata: &Option<ColMetadataList>,
    value: &Value<&str>,
) -> Result<(), InsertDataError> {
    let col_ftype = new_metadata
        .as_ref()
        .ok_or_else(|| {
            InsertDataError::NewColumnNotInColumnMetadata(value.col_ix.into())
        })?
        .get(value.col_ix)
        .ok_or_else(|| {
            InsertDataError::NewColumnNotInColumnMetadata(value.col_ix.into())
        })
        .map(|(_, md)| FType::from_coltype(&md.coltype))?;

    let (is_compat, compat_info) = col_ftype.datum_compatible(&value.value);

    let bad_continuous_value = match value.value {
        Datum::Continuous(ref x) => !x.is_finite(),
        _ => false,
    };

    if is_compat {
        if bad_continuous_value {
            Err(InsertDataError::NonFiniteContinuousValue {
                col: value.col_ix.to_owned(),
                value: value.value.to_f64_opt().unwrap(),
            })
        } else {
            Ok(())
        }
    } else {
        Err(InsertDataError::DatumIncompatibleWithColumn {
            col: value.col_ix.to_owned(),
            ftype: compat_info.ftype,
            ftype_req: compat_info.ftype_req,
        })
    }
}

fn validate_row_values<R: RowIndex, C: ColumnIndex>(
    row: &Row<R, C>,
    row_ix: usize,
    row_exists: bool,
    col_metadata: &Option<ColMetadataList>,
    col_ix_lookup: &Option<HashMap<&str, usize>>,
    insert_tasks: &mut InsertDataTasks,
    engine: &Engine,
) -> Result<IndexRow, InsertDataError> {
    let n_cols = engine.n_cols();

    let mut index_row = IndexRow {
        row_ix,
        values: vec![],
    };

    row.values.iter().try_for_each(|value| {
        match value.col_ix.col_ix(engine.codebook()) {
            Ok(col_ix) => {
                // check whether the datum is missing.
                if row_exists {
                    if engine.datum(row_ix, col_ix).unwrap().is_missing() {
                        insert_tasks.overwrite_missing = true;
                    } else {
                        insert_tasks.overwrite_present = true;
                    }
                }

                // determine whether the value is compatible
                // with the FType of the column
                let ftype_compat = engine
                    .ftype(col_ix)
                    .unwrap()
                    .datum_compatible(&value.value);

                let bad_continuous_value = match value.value {
                    Datum::Continuous(ref x) => !x.is_finite(),
                    _ => false,
                };

                if ftype_compat.0 {
                    if bad_continuous_value {
                        let col = &engine.codebook.col_metadata[col_ix].name;
                        Err(InsertDataError::NonFiniteContinuousValue {
                            col: col.clone(),
                            value: value.value.to_f64_opt().unwrap(),
                        })
                    } else {
                        Ok(col_ix)
                    }
                } else {
                    let col = &engine.codebook.col_metadata[col_ix].name;
                    Err(InsertDataError::DatumIncompatibleWithColumn {
                        col: col.clone(),
                        ftype_req: ftype_compat.1.ftype_req,
                        ftype: ftype_compat.1.ftype,
                    })
                }
            }
            Err(_) => {
                value
                    .col_ix
                    .col_str()
                    .ok_or_else(|| {
                        InsertDataError::IntegerIndexNewColumn(
                            value
                                .col_ix
                                .col_usize()
                                .expect("Column index does not have a string or usize representation")
                        )
                    })
                    .and_then(|name| {
                        // TODO: get rid of this clone
                        let new_val = Value {
                            col_ix: name,
                            value: value.value.clone(),
                        };
                        validate_new_col_ftype(col_metadata, &new_val).and_then(
                            |_| {
                                insert_tasks.new_cols.insert(name.to_owned());
                                col_ix_from_lookup(name, col_ix_lookup)
                                    .map(|ix| ix + n_cols)
                            },
                        )
                    })
            }
        }
        .map(|col_ix| {
            index_row.values.push(IndexValue {
                col_ix,
                value: value.value.clone(),
            });
        })
    })?;
    Ok(index_row)
}

/// Get a summary of the tasks required to insert `rows` into `Engine`.
pub(crate) fn insert_data_tasks<R: RowIndex, C: ColumnIndex>(
    rows: &[Row<R, C>],
    col_metadata: &Option<ColMetadataList>,
    engine: &Engine,
) -> Result<(InsertDataTasks, Vec<IndexRow>), InsertDataError> {
    const EXISTING_ROW: bool = true;
    const NEW_ROW: bool = false;

    // Get a map into the new column indices if they exist
    let col_ix_lookup = ix_lookup_from_codebook(col_metadata);

    // Get a list of all the row names. The row names must be included in the
    // codebook in order to insert data.
    let n_rows = engine.n_rows();

    let mut tasks = InsertDataTasks::new();

    let index_rows: Vec<IndexRow> = rows
        .iter()
        .map(|row| match row.row_ix.row_ix(engine.codebook()) {
            Ok(row_ix) => {
                if row.is_empty() {
                    let name = engine.codebook.row_names.name(row_ix).unwrap();
                    Err(InsertDataError::EmptyRow(name.clone()))
                } else {
                    validate_row_values(
                        row,
                        row_ix,
                        EXISTING_ROW,
                        col_metadata,
                        &col_ix_lookup,
                        &mut tasks,
                        engine,
                    )
                }
            }
            Err(_) => {
                // row index is either out of bounds or does not exist in the codebook
                if row.is_empty() {
                    Err(InsertDataError::EmptyRow(format!("{:?}", row.row_ix)))
                } else {
                    validate_row_values(
                        row,
                        {
                            let n = tasks.new_rows.len();
                            row.row_ix
                                .row_str()
                                .ok_or_else(|| {
                                    let ix = row
                                        .row_ix
                                        .row_usize()
                                        .expect("Index doesn't have a string or usize representation");
                                    InsertDataError::IntegerIndexNewRow(ix)
                                })
                                .map(|row_name| {
                                    tasks
                                        .new_rows
                                        .insert(String::from(row_name));
                                })?;
                            n_rows + n
                        },
                        NEW_ROW,
                        col_metadata,
                        &col_ix_lookup,
                        &mut tasks,
                        engine,
                    )
                }
            }
        })
        .collect::<Result<Vec<IndexRow>, InsertDataError>>()?;
    Ok((tasks, index_rows))
}

pub(crate) fn maybe_add_categories<R: RowIndex, C: ColumnIndex>(
    rows: &[Row<R, C>],
    engine: &mut Engine,
    mode: WriteMode,
) -> Result<Vec<SupportExtension>, InsertDataError> {
    let mut extended_value_map: HashMap<usize, ValueMapExtension> =
        HashMap::new();

    // This code gets all the supports for all the categorical columns for
    // which data are to be inserted.
    // For each value (cell) in each row...
    rows.iter().try_for_each(|row| {
        row.values.iter().try_for_each(|new_value| {
            // if the column is categorical, see if we need to add support,
            // otherwise carry on.
            match new_value.col_ix.col_ix(engine.codebook()) {
                Err(_) => Ok(()), // IndexError means new column
                Ok(col_ix) => {
                    let col_metadata = &engine.codebook.col_metadata[col_ix];
                    let col_name = col_metadata.name.as_str();

                    match &col_metadata.coltype {
                        ColType::Categorical { value_map, .. } => {
                            match (&new_value.value, value_map) {
                                (Datum::Categorical(Category::String(x)), ValueMap::String(vm)) => {
                                    if !vm.contains_cat(x) {
                                        let ext_vm = extended_value_map.entry(col_ix).or_insert_with(ValueMapExtension::new_string);
                                        ext_vm.extend(Category::String(x.clone())).map_err(
                                            |e| match e {
                                                ValueMapExtensionError::ExtensionOfDifferingType(a, b) => {
                                                    InsertDataError::WrongCategoryAndType(a, b, col_name.to_string())
                                                }
                                            })?;
                                    };
                                    Ok(())
                                },
                                (Datum::Categorical(Category::UInt(x)), ValueMap::UInt(old_max)) => {
                                    if *old_max as u32 <= *x {
                                        let ext_max = extended_value_map.entry(col_ix).or_insert_with(ValueMapExtension::new_uint);
                                        ext_max.extend(Category::UInt(*x)).map_err(
                                            |e| match e {
                                                ValueMapExtensionError::ExtensionOfDifferingType(a, b) => {
                                                    InsertDataError::WrongCategoryAndType(a, b, col_name.to_string())
                                                }
                                            })?;
                                    }
                                    Ok(())
                                },
                                (Datum::Missing, _) |
                                (Datum::Categorical(Category::Bool(_)), ValueMap::Bool) => {
                                    Ok(())
                                },
                                _ => {
                                    Err(
                                    InsertDataError::DatumIncompatibleWithColumn {
                                        col: (*col_name).into(),
                                        ftype_req: FType::Categorical,
                                        // this should never fail because TryFrom only
                                        // fails for Datum::Missing, and that case is
                                        // handled above
                                        ftype: (&new_value.value).try_into().unwrap(),
                                    },
                                )}
                            }
                        }
                        _ => Ok(())
                    }
                }
            }
        })
    })?;

    if !mode.allow_extend_support && !extended_value_map.is_empty() {
        // support extension not allowed
        return Err(InsertDataError::ModeForbidsCategoryExtension);
    }

    let mut cols_extended: Vec<SupportExtension> = Vec::new();

    // Here we loop through all the categorical insertions generated above and
    // determine whether we need to extend categorical support by comparing the
    // existing support (n_cats, or k) for each column with the maximum value
    // requested to be inserted into that column. If the value exceeds the
    // support of that column, we extend the support.
    for (col_ix, value_map_extension) in extended_value_map.drain() {
        incr_column_categories(engine, col_ix, &value_map_extension)?;
        cols_extended.push(SupportExtension::Categorical {
            col_ix,
            col_name: engine.codebook.col_metadata[col_ix].name.clone(),
            value_map_extension,
        })
    }

    Ok(cols_extended)
}

fn incr_category_in_codebook(
    codebook: &mut Codebook,
    col_ix: usize,
    value_map_extension: &ValueMapExtension,
) -> Result<(), InsertDataError> {
    let col_name = codebook.col_metadata[col_ix].name.clone();
    match codebook.col_metadata[col_ix].coltype {
        ColType::Categorical {
            ref mut k,
            ref mut value_map,
            ..
        } => {
            // TODO: Does this capture any errors from a user or just errors from within lace?
            value_map.extend(value_map_extension.clone()).map_err(
                |e| match e {
                    ValueMapExtensionError::ExtensionOfDifferingType(a, b) => {
                        InsertDataError::WrongCategoryAndType(a, b, col_name)
                    }
                },
            )?;
            *k = value_map.len();
            Ok(())
        }
        _ => panic!("Tried to change cardinality of non-categorical column"),
    }
}

fn incr_column_categories(
    engine: &mut Engine,
    col_ix: usize,
    extended_value_map: &ValueMapExtension,
) -> Result<(), InsertDataError> {
    // Adjust in codebook
    incr_category_in_codebook(
        &mut engine.codebook,
        col_ix,
        extended_value_map,
    )?;

    let n_cats_req = match engine.codebook.col_metadata[col_ix].coltype {
        ColType::Categorical { k, .. } => k,
        _ => panic!("Requested non-categorical column"),
    };

    // Adjust component models, priors, suffstats
    engine.states.iter_mut().for_each(|state| {
        match state.feature_mut(col_ix) {
            ColModel::Categorical(column) => {
                column.prior = SymmetricDirichlet::new_unchecked(
                    column.prior.alpha(),
                    n_cats_req,
                );
                column.components.iter_mut().for_each(|cpnt| {
                    cpnt.stat = CategoricalSuffStat::from_parts_unchecked(
                        cpnt.stat.n(),
                        {
                            let mut counts = cpnt.stat.counts().clone();
                            counts.resize(n_cats_req, 0.0);
                            counts
                        },
                    );

                    cpnt.fx = Categorical::new_unchecked({
                        let mut ln_weights = cpnt.fx.ln_weights().clone();
                        ln_weights.resize(n_cats_req, f64::NEG_INFINITY);
                        ln_weights
                    });
                })
            }
            _ => panic!("Requested non-categorical column"),
        }
    });
    Ok(())
}

macro_rules! new_col_arm {
    (
        $coltype: ident,
        $htype: ty,
        $errvar: ident,
        $colmd: ident,
        $hyper: ident,
        $prior: ident,
        $n_rows: ident,
        $id: ident,
        $xtype: ty,
        $rng: ident
    ) => {{
        let data: SparseContainer<$xtype> =
            SparseContainer::all_missing($n_rows);

        match ($hyper, $prior) {
            (Some(h), _) => {
                let pr = if let Some(pr) = $prior {
                    pr.clone()
                } else {
                    h.draw(&mut $rng)
                };
                let column = Column::new($id, data, pr, h.clone());
                Ok(ColModel::$coltype(column))
            }
            (None, Some(pr)) => {
                // use a dummy hyper, we're going to ignore it
                let mut column =
                    Column::new($id, data, pr.clone(), <$htype>::default());
                column.ignore_hyper = true;
                Ok(ColModel::$coltype(column))
            }
            (None, None) => Err(InsertDataError::NoGaussianHyperForNewColumn(
                $colmd.name.clone(),
            )),
        }
    }};
}

pub(crate) fn create_new_columns<R: rand::Rng>(
    col_metadata: &ColMetadataList,
    state_shape: (usize, usize),
    mut rng: &mut R,
) -> Result<Vec<ColModel>, InsertDataError> {
    let (n_rows, n_cols) = state_shape;
    col_metadata
        .iter()
        .enumerate()
        .map(|(i, colmd)| {
            let id = i + n_cols;
            match &colmd.coltype {
                ColType::Continuous { hyper, prior } => new_col_arm!(
                    Continuous,
                    crate::stats::prior::nix::NixHyper,
                    NoGaussianHyperForNewColumn,
                    colmd,
                    hyper,
                    prior,
                    n_rows,
                    id,
                    f64,
                    rng
                ),
                ColType::Count { hyper, prior } => new_col_arm!(
                    Count,
                    crate::stats::prior::pg::PgHyper,
                    NoPoissonHyperForNewColumn,
                    colmd,
                    hyper,
                    prior,
                    n_rows,
                    id,
                    u32,
                    rng
                ),
                ColType::Categorical {
                    k, hyper, prior, ..
                } => {
                    let data: SparseContainer<u32> =
                        SparseContainer::all_missing(n_rows);

                    let id = i + n_cols;
                    match (hyper, prior) {
                        (Some(h), _) => {
                            let pr = if let Some(pr) = prior {
                                pr.clone()
                            } else {
                                h.draw(*k, &mut rng)
                            };
                            let column = Column::new(id, data, pr, h.clone());
                            Ok(ColModel::Categorical(column))
                        }
                        (None, Some(pr)) => {
                            use crate::stats::prior::csd::CsdHyper;
                            let mut column = Column::new(
                                id,
                                data,
                                pr.clone(),
                                CsdHyper::default(),
                            );
                            column.ignore_hyper = true;
                            Ok(ColModel::Categorical(column))
                        }
                        (None, None) => Err(
                            InsertDataError::NoCategoricalHyperForNewColumn(
                                colmd.name.clone(),
                            ),
                        ),
                    }
                }
            }
        })
        .collect()
}

pub(crate) fn remove_cell(engine: &mut Engine, row_ix: usize, col_ix: usize) {
    engine.states.iter_mut().for_each(|state| {
        state.remove_datum(row_ix, col_ix);
    })
}

pub(crate) fn remove_col(engine: &mut Engine, col_ix: usize) {
    // remove the column from the codebook and re-index
    engine.codebook.col_metadata.remove_by_index(col_ix);
    let mut rng = engine.rng.clone();
    engine.states.iter_mut().for_each(|state| {
        // deletes the column and re-indexes
        state.del_col(col_ix, &mut rng);
    });
}

pub(crate) fn check_if_removes_col(
    engine: &Engine,
    rm_rows: &BTreeSet<usize>,
    mut rm_cell_cols: HashMap<usize, i64>,
) -> BTreeSet<usize> {
    let mut to_rm: BTreeSet<usize> = BTreeSet::new();
    // rm_cell_cols.values_mut().for_each(|val| {*val -= rm_rows.len() as i64});
    rm_cell_cols.iter_mut().for_each(|(col_ix, val)| {
        let mut present_count = 0_i64;
        let mut remove = true;
        for row_ix in 0..engine.n_rows() {
            if present_count > *val {
                remove = false;
                break;
            }
            if !rm_rows.contains(&row_ix)
                && !engine.datum(row_ix, *col_ix).unwrap().is_missing()
            {
                present_count += 1;
            }
        }
        if remove {
            to_rm.insert(*col_ix);
        }
    });
    to_rm
}

pub(crate) fn check_if_removes_row(
    engine: &Engine,
    rm_cols: &BTreeSet<usize>,
    mut rm_cell_rows: HashMap<usize, i64>,
) -> BTreeSet<usize> {
    let mut to_rm: BTreeSet<usize> = BTreeSet::new();
    rm_cell_rows.iter_mut().for_each(|(row_ix, val)| {
        let mut present_count = 0_i64;
        let mut remove = true;
        for col_ix in 0..engine.n_cols() {
            if present_count > *val {
                remove = false;
                break;
            }
            if !rm_cols.contains(&col_ix)
                && !engine.datum(*row_ix, col_ix).unwrap().is_missing()
            {
                present_count += 1;
            }
        }
        if remove {
            to_rm.insert(*row_ix);
        }
    });
    to_rm
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::codebook::ColMetadata;
    use crate::codebook::ColType;
    use crate::codebook::ValueMap;
    use crate::data::data_source;
    use crate::stats::prior::csd::CsdHyper;

    #[cfg(feature = "examples")]
    mod requiring_examples {
        use super::*;
        use crate::examples::Example;

        #[test]
        fn errors_when_no_col_metadata_when_new_columns() {
            let engine = Example::Animals.engine().unwrap();
            let moose_updates = Row::<String, String> {
                row_ix: "moose".into(),
                values: vec![
                    Value {
                        col_ix: "does+taxes".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };

            let result = insert_data_tasks(&[moose_updates], &None, &engine);

            assert!(result.is_err());
            match result {
                Err(InsertDataError::NewColumnNotInColumnMetadata(s)) => {
                    assert_eq!(s, String::from("does+taxes"))
                }
                Err(err) => panic!("wrong error: {:?}", err),
                Ok(_) => panic!("failed to fail"),
            }
        }

        #[test]
        fn errors_when_new_column_not_in_col_metadata() {
            let engine = Example::Animals.engine().unwrap();
            let moose_updates = Row::<String, String> {
                row_ix: "moose".into(),
                values: vec![
                    Value {
                        col_ix: "does+taxes".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };

            let col_metadata = ColMetadataList::new(vec![ColMetadata {
                name: "dances".into(),
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    prior: None,
                    value_map: ValueMap::UInt(2),
                },
                notes: None,
                missing_not_at_random: false,
            }])
            .unwrap();

            let result = insert_data_tasks(
                &[moose_updates],
                &Some(col_metadata),
                &engine,
            );

            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err(),
                InsertDataError::NewColumnNotInColumnMetadata(
                    "does+taxes".into()
                )
            );
        }

        #[test]
        fn tasks_on_one_existing_row() {
            let engine = Example::Animals.engine().unwrap();
            let moose_updates = Row::<String, String> {
                row_ix: "moose".into(),
                values: vec![
                    Value {
                        col_ix: "swims".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };
            let rows = vec![moose_updates];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &None, &engine).unwrap();

            assert!(tasks.new_rows.is_empty());
            assert!(tasks.new_cols.is_empty());
            assert!(!tasks.overwrite_missing);
            assert!(tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![IndexRow {
                    row_ix: 15,
                    values: vec![
                        IndexValue {
                            col_ix: 36,
                            value: Datum::Categorical(1_u32.into())
                        },
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(1_u32.into())
                        },
                    ]
                }]
            );
        }

        #[test]
        fn tasks_on_one_new_row() {
            let engine = Example::Animals.engine().unwrap();
            let pegasus = Row::<String, String> {
                row_ix: "pegasus".into(),
                values: vec![
                    Value {
                        col_ix: "swims".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };
            let rows = vec![pegasus];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &None, &engine).unwrap();

            assert_eq!(tasks.new_rows.len(), 1);
            assert!(tasks.new_rows.contains("pegasus"));
            assert!(tasks.new_cols.is_empty());
            assert!(!tasks.overwrite_missing);
            assert!(!tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![IndexRow {
                    row_ix: 50,
                    values: vec![
                        IndexValue {
                            col_ix: 36,
                            value: Datum::Categorical(1_u32.into())
                        },
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(1_u32.into())
                        },
                    ]
                }]
            );
        }

        #[test]
        fn tasks_on_two_new_rows() {
            let engine = Example::Animals.engine().unwrap();
            let pegasus = Row::<String, String> {
                row_ix: "pegasus".into(),
                values: vec![
                    Value {
                        col_ix: "swims".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };

            let man = Row::<String, String> {
                row_ix: "man".into(),
                values: vec![
                    Value {
                        col_ix: "smart".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "hunter".into(),
                        value: Datum::Categorical(0_u32.into()),
                    },
                ],
            };
            let rows = vec![pegasus, man];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &None, &engine).unwrap();

            assert_eq!(tasks.new_rows.len(), 2);
            assert!(tasks.new_rows.contains("pegasus"));
            assert!(tasks.new_rows.contains("man"));

            assert!(tasks.new_cols.is_empty());
            assert!(!tasks.overwrite_missing);
            assert!(!tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![
                    IndexRow {
                        row_ix: 50,
                        values: vec![
                            IndexValue {
                                col_ix: 36,
                                value: Datum::Categorical(1_u32.into())
                            },
                            IndexValue {
                                col_ix: 34,
                                value: Datum::Categorical(1_u32.into())
                            },
                        ]
                    },
                    IndexRow {
                        row_ix: 51,
                        values: vec![
                            IndexValue {
                                col_ix: 80,
                                value: Datum::Categorical(1_u32.into())
                            },
                            IndexValue {
                                col_ix: 58,
                                value: Datum::Categorical(0_u32.into())
                            },
                        ]
                    }
                ]
            );
        }

        #[test]
        fn tasks_on_one_new_and_one_existing_row() {
            let engine = Example::Animals.engine().unwrap();
            let pegasus = Row::<String, String> {
                row_ix: "pegasus".into(),
                values: vec![
                    Value {
                        col_ix: "swims".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };

            let moose = Row::<String, String> {
                row_ix: "moose".into(),
                values: vec![
                    Value {
                        col_ix: "smart".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "hunter".into(),
                        value: Datum::Categorical(0_u32.into()),
                    },
                ],
            };
            let rows = vec![pegasus, moose];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &None, &engine).unwrap();

            assert_eq!(tasks.new_rows.len(), 1);
            assert!(tasks.new_rows.contains("pegasus"));

            assert!(tasks.new_cols.is_empty());
            assert!(!tasks.overwrite_missing);
            assert!(tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![
                    IndexRow {
                        row_ix: 50,
                        values: vec![
                            IndexValue {
                                col_ix: 36,
                                value: Datum::Categorical(1_u32.into())
                            },
                            IndexValue {
                                col_ix: 34,
                                value: Datum::Categorical(1_u32.into())
                            },
                        ]
                    },
                    IndexRow {
                        row_ix: 15,
                        values: vec![
                            IndexValue {
                                col_ix: 80,
                                value: Datum::Categorical(1_u32.into())
                            },
                            IndexValue {
                                col_ix: 58,
                                value: Datum::Categorical(0_u32.into())
                            },
                        ]
                    }
                ]
            );
        }

        #[test]
        fn tasks_on_one_new_col_in_existing_row() {
            let engine = Example::Animals.engine().unwrap();
            let col_metadata = ColMetadataList::new(vec![ColMetadata {
                name: "dances".into(),
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    prior: None,
                    value_map: ValueMap::UInt(2),
                },
                notes: None,
                missing_not_at_random: false,
            }])
            .unwrap();
            let moose_updates = Row::<String, String> {
                row_ix: "moose".into(),
                values: vec![
                    Value {
                        col_ix: "dances".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };
            let rows = vec![moose_updates];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &Some(col_metadata), &engine).unwrap();

            assert!(tasks.new_rows.is_empty());
            assert_eq!(tasks.new_cols.len(), 1);
            assert!(tasks.new_cols.contains("dances"));

            assert!(!tasks.overwrite_missing);
            assert!(tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![IndexRow {
                    row_ix: 15,
                    values: vec![
                        IndexValue {
                            col_ix: 85,
                            value: Datum::Categorical(1_u32.into())
                        },
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(1_u32.into())
                        },
                    ]
                }]
            );
        }

        #[test]
        fn tasks_on_one_new_col_in_new_row() {
            let engine = Example::Animals.engine().unwrap();

            let col_metadata = ColMetadataList::new(vec![ColMetadata {
                name: "dances".into(),
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    prior: None,
                    value_map: ValueMap::UInt(2),
                },
                notes: None,
                missing_not_at_random: false,
            }])
            .unwrap();

            let peanut = Row::<String, String> {
                row_ix: "peanut".into(),
                values: vec![
                    Value {
                        col_ix: "dances".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(0_u32.into()),
                    },
                ],
            };
            let rows = vec![peanut];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &Some(col_metadata), &engine).unwrap();

            assert_eq!(tasks.new_rows.len(), 1);
            assert!(tasks.new_rows.contains("peanut"));

            assert_eq!(tasks.new_cols.len(), 1);
            assert!(tasks.new_cols.contains("dances"));

            assert!(!tasks.overwrite_missing);
            assert!(!tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![IndexRow {
                    row_ix: 50,
                    values: vec![
                        IndexValue {
                            col_ix: 85,
                            value: Datum::Categorical(1_u32.into())
                        },
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(0_u32.into())
                        },
                    ]
                }]
            );
        }

        #[test]
        fn tasks_on_two_new_cols_in_existing_row() {
            let engine = Example::Animals.engine().unwrap();
            let col_metadata = ColMetadataList::new(vec![
                ColMetadata {
                    name: "dances".into(),
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        prior: None,
                        value_map: ValueMap::UInt(2),
                    },
                    notes: None,
                    missing_not_at_random: false,
                },
                ColMetadata {
                    name: "eats+figs".into(),
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        prior: None,
                        value_map: ValueMap::UInt(2),
                    },
                    notes: None,
                    missing_not_at_random: false,
                },
            ])
            .unwrap();

            let moose_updates = Row::<String, String> {
                row_ix: "moose".into(),
                values: vec![
                    Value {
                        col_ix: "flys".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                    Value {
                        col_ix: "eats+figs".into(),
                        value: Datum::Categorical(0_u32.into()),
                    },
                    Value {
                        col_ix: "dances".into(),
                        value: Datum::Categorical(1_u32.into()),
                    },
                ],
            };
            let rows = vec![moose_updates];
            let (tasks, ixrows) =
                insert_data_tasks(&rows, &Some(col_metadata), &engine).unwrap();

            assert!(tasks.new_rows.is_empty());
            assert_eq!(tasks.new_cols.len(), 2);
            assert!(tasks.new_cols.contains("dances"));
            assert!(tasks.new_cols.contains("eats+figs"));

            assert!(!tasks.overwrite_missing);
            assert!(tasks.overwrite_present);

            assert_eq!(
                ixrows,
                vec![IndexRow {
                    row_ix: 15,
                    values: vec![
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(1_u32.into())
                        },
                        IndexValue {
                            col_ix: 86,
                            value: Datum::Categorical(0_u32.into())
                        },
                        IndexValue {
                            col_ix: 85,
                            value: Datum::Categorical(1_u32.into())
                        },
                    ]
                }]
            );
        }
    }

    fn quick_codebook() -> Codebook {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            prior: None,
            value_map: ValueMap::try_from(vec![
                String::from("red"),
                String::from("green"),
            ])
            .unwrap(),
        };
        let md0 = ColMetadata {
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
            missing_not_at_random: false,
        };
        let md1 = ColMetadata {
            name: "1".to_string(),
            coltype,
            notes: None,
            missing_not_at_random: false,
        };
        let md2 = ColMetadata {
            name: "2".to_string(),
            coltype: ColType::Categorical {
                k: 3,
                hyper: None,
                prior: None,
                value_map: ValueMap::UInt(3),
            },
            notes: None,
            missing_not_at_random: false,
        };

        let col_metadata = ColMetadataList::new(vec![md0, md1, md2]).unwrap();
        Codebook::new("table".to_string(), col_metadata)
    }

    #[test]
    fn incr_cats_in_codebook_without_suppl_metadata_for_no_valmap_col() {
        let mut codebook = quick_codebook();

        let n_cats_before = match &codebook.col_metadata[2].coltype {
            ColType::Categorical {
                k,
                value_map: ValueMap::UInt(3),
                ..
            } => *k,
            ColType::Categorical { value_map, .. } => {
                panic!(
                    "starting value_map should have been U32(3), was {:?}",
                    value_map
                );
            }
            _ => panic!("should've been categorical"),
        };

        assert_eq!(n_cats_before, 3);

        let mut extension = ValueMapExtension::new_uint();
        extension.extend(Category::UInt(3)).unwrap();

        let result = incr_category_in_codebook(&mut codebook, 2, &extension);
        result.unwrap();

        let n_cats_after = match &codebook.col_metadata[2].coltype {
            ColType::Categorical {
                k,
                value_map: ValueMap::UInt(4),
                ..
            } => *k,
            ColType::Categorical { value_map, .. } => {
                panic!("value_map should be U32(4), was: {:?}", value_map)
            }
            _ => panic!("should've been categorical"),
        };

        assert_eq!(n_cats_after, 4);
    }

    #[test]
    fn incr_cats_in_codebook_with_suppl_metadata_for_valmap_col() {
        let mut codebook = quick_codebook();

        match &codebook.col_metadata[0].coltype {
            ColType::Categorical {
                k, value_map: vm, ..
            } => {
                assert_eq!(*k, 2);
                assert_eq!(vm.len(), 2);
            }
            _ => panic!("should've been categorical with valmap"),
        };

        let mut extension = ValueMapExtension::new_string();
        extension
            .extend(Category::String("blue".to_string()))
            .unwrap();

        let result = incr_category_in_codebook(&mut codebook, 0, &extension);

        assert!(result.is_ok());

        match &codebook.col_metadata[0].coltype {
            ColType::Categorical {
                k, value_map: vm, ..
            } => {
                assert_eq!(vm.len(), 3);
                assert_eq!(*k, 3);
            }
            _ => panic!("should've been categorical with valmap"),
        };
    }

    #[test]
    fn append_bool() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: Some(CsdHyper::default()),
            prior: None,
            value_map: ValueMap::Bool,
        };
        let md0 = ColMetadata {
            name: "bool_col".to_string(),
            coltype: coltype.clone(),
            notes: None,
            missing_not_at_random: false,
        };

        let mut engine = Engine::new(
            1,
            Codebook::new(
                "test".to_string(),
                ColMetadataList::new(vec![]).unwrap(),
            ),
            data_source::DataSource::Empty,
            0,
            rand_xoshiro::Xoshiro256Plus::seed_from_u64(0x1234),
        )
        .unwrap();

        // Insert once with specific metadata.
        engine
            .insert_data(
                vec![(
                    "abc",
                    vec![(
                        "bool_col",
                        Datum::Categorical(Category::Bool(false)),
                    )],
                )
                    .into()],
                Some(ColMetadataList::new(vec![md0]).unwrap()),
                WriteMode::unrestricted(),
            )
            .unwrap();

        // Insert again without metadata for the bool column.
        engine
            .insert_data(
                vec![(
                    "def",
                    vec![(
                        "bool_col",
                        Datum::Categorical(Category::Bool(false)),
                    )],
                )
                    .into()],
                None,
                WriteMode::unrestricted(),
            )
            .unwrap();
    }
}

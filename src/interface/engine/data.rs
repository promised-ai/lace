use super::error::InsertDataError;
use crate::cc::ColModel;
use crate::cc::Column;
use crate::cc::FType;
use crate::{Engine, OracleT};

use braid_codebook::ColMetadataList;
use braid_codebook::ColType;
use braid_data::SparseContainer;
use braid_stats::labeler::{Label, LabelerPrior};
use braid_stats::prior::{Csd, Ng, Pg};
use braid_stats::Datum;

use indexmap::IndexSet;
use rv::data::CategoricalSuffStat;
use rv::dist::{Categorical, SymmetricDirichlet};
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::f64::NEG_INFINITY;

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

/// Defines how/where data may be inserted, which day may and may not be
/// overwritten, and whether data may extend the domain
///
/// # Example
///
/// Default `WriteMode` only allows appending supported values to new rows or
/// columns
/// ```
/// use braid::{WriteMode, InsertMode, OverwriteMode};
/// let mode_new = WriteMode::new();
/// let mode_def = WriteMode::default();
///
/// assert_eq!(
///     mode_new,
///     WriteMode {
///         insert: InsertMode::Unrestricted,
///         overwrite: OverwriteMode::Deny,
///         allow_extend_support: false,
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
    pub allow_extend_support: bool,
}

impl WriteMode {
    /// Allows new data to be appended only to new rows/columns. No overwriting
    /// and no support extension.
    #[inline]
    pub fn new() -> WriteMode {
        WriteMode {
            insert: InsertMode::Unrestricted,
            overwrite: OverwriteMode::Deny,
            allow_extend_support: false,
        }
    }

    #[inline]
    pub fn unrestricted() -> WriteMode {
        WriteMode {
            insert: InsertMode::Unrestricted,
            overwrite: OverwriteMode::Allow,
            allow_extend_support: true,
        }
    }
}

impl Default for WriteMode {
    fn default() -> WriteMode {
        WriteMode::new()
    }
}

/// A datum for insertion into a certain column
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Value {
    /// Name of the column
    pub col_name: String,
    /// The value of the cell
    pub value: Datum,
}

impl<S: Into<String>> From<(S, Datum)> for Value {
    fn from(value: (S, Datum)) -> Value {
        Value {
            col_name: value.0.into(),
            value: value.1,
        }
    }
}

/// A list of data for insertion into a certain row
///
/// # Example
///
/// ```
/// # use braid::Row;
/// use braid::Value;
/// use braid_stats::Datum;
///
/// let row = Row {
///     row_name: String::from("vampire"),
///     values: vec![
///         Value {
///             col_name: String::from("sucks_blood"),
///             value: Datum::Categorical(1),
///         },
///         Value {
///             col_name: String::from("drinks_wine"),
///             value: Datum::Categorical(0),
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
/// # use braid::Row;
/// # use braid_stats::Datum;
/// let row: Row = (
///     "vampire",
///     vec![
///         ("sucks_blood", Datum::Categorical(1)),
///         ("drinks_wine", Datum::Categorical(0)),
///     ]
/// ).into();
///
/// assert_eq!(row.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// The name of the row
    pub row_name: String,
    /// The cells and values to fill in
    pub values: Vec<Value>,
}

impl<Sr, Sc> From<(Sr, Vec<(Sc, Datum)>)> for Row
where
    Sr: Into<String>,
    Sc: Into<String>,
{
    fn from(mut row: (Sr, Vec<(Sc, Datum)>)) -> Row {
        Row {
            row_name: row.0.into(),
            values: row.1.drain(..).map(Value::from).collect(),
        }
    }
}

impl<S: Into<String>> From<(S, Vec<Value>)> for Row {
    fn from(row: (S, Vec<Value>)) -> Row {
        Row {
            row_name: row.0.into(),
            values: row.1,
        }
    }
}

impl Row {
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

// Because braid uses integer indices for rows and columns
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

/// A summary of the tasks required to insert certain data into an `Engine`
#[derive(Debug)]
pub struct InsertDataTasks {
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
    pub fn validate_insert_mode(
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
fn ix_lookup_from_codebook<'a>(
    col_metadata: &'a Option<ColMetadataList>,
) -> Option<HashMap<&'a str, usize>> {
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
            .map(|col| *col),
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
                let shape = (engine.nrows(), engine.ncols());
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
                        // XXX: if a panic happens here its our fault.
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

/// Get a summary of the tasks required to insert `rows` into `Engine`.
pub(crate) fn insert_data_tasks(
    rows: &[Row],
    col_metadata: &Option<ColMetadataList>,
    engine: &Engine,
) -> Result<(InsertDataTasks, Vec<IndexRow>), InsertDataError> {
    // Get a map into the new column indices if they exist
    let ix_lookup = ix_lookup_from_codebook(&col_metadata);

    // Get a list of all the row names. The row names must be included in the
    // codebook in order to insert data.
    // let empty = vec![];
    let row_names = &engine.codebook.row_names;

    let mut index_rows: Vec<IndexRow> = Vec::new();
    let mut nrows = engine.nrows();
    let ncols = engine.ncols();

    let (new_rows, new_cols, overwrite_missing, overwrite_present) = {
        let mut overwrite_missing = false;
        let mut overwrite_present = false;
        let mut new_rows: IndexSet<String> = IndexSet::new();
        let mut new_cols: HashSet<String> = HashSet::new();

        rows.iter().try_for_each(|row| {
            if !new_rows.contains(&row.row_name)
                && row_names.index(&row.row_name).is_none()
            {
                if row.is_empty() {
                    return Err(InsertDataError::EmptyRow(row.row_name.clone()))
                }

                // If the row does not exist..
                let mut index_row = IndexRow {
                    row_ix: nrows,
                    values: vec![],
                };
                nrows += 1;
                // Add the row name to the list of new rows
                new_rows.insert(row.row_name.clone());

                row.values
                    .iter()
                    .try_for_each(|value| {
                        let col = &value.col_name;
                        let colmd = engine.codebook.col_metadata.get(col);

                        // check whether the column is new
                        if !new_cols.contains(col) && colmd.is_none() {
                            new_cols.insert(col.to_owned());
                        }

                        match colmd {
                            // If the column exists
                            Some((col_ix, _)) => {
                                // Check whether the value to be inserted is
                                // compatible with the FType of the cell
                                let ftype_compat = engine
                                    .ftype(col_ix)
                                    .unwrap()
                                    .datum_compatible(&value.value);

                                if  ftype_compat.0 {
                                    Ok(col_ix)
                                } else {
                                    Err(InsertDataError::DatumIncompatibleWithColumn{
                                            col: col.to_owned(),
                                            ftype_req: ftype_compat.1.ftype_req,
                                            ftype: ftype_compat.1.ftype,
                                    })
                                }
                            },
                            // If the column doesn't exist, get the col_ixs
                            // from the col_metadata lookup
                            None => col_ix_from_lookup(col, &ix_lookup)
                                .map(|ix| ix + ncols),
                        }
                        .map(|col_ix| {
                            // create the index value
                            index_row.values.push(IndexValue {
                                col_ix,
                                value: value.value.clone(),
                            });
                        })
                    })
                    .map(|_| {
                        index_rows.push(index_row);
                    })
            } else {
                // If this row exists...
                // Get the row index by enumerating the row names. 
                // TODO: optimize away linear row name lookup
                let row_ix = row_names.index(&row.row_name)
                    .expect("Unable to get row index");

                let mut index_row = IndexRow {
                    row_ix,
                    values: vec![],
                };

                row.values
                    .iter()
                    .try_for_each(|value| {
                        let col = &value.col_name;
                        let colmd = engine.codebook.col_metadata.get(col);

                        match colmd {
                            // if this is a existing cell
                            Some((col_ix, _)) => {
                                // check whether the datum is missing.
                                if engine
                                    .datum(row_ix, col_ix)
                                    .unwrap()
                                    .is_missing()
                                {
                                    overwrite_missing = true;
                                } else {
                                    overwrite_present = true;
                                }

                                // determine whether the value is compatible
                                // with the FType of the column
                                let ftype_compat = engine
                                    .ftype(col_ix)
                                    .unwrap()
                                    .datum_compatible(&value.value);

                                if ftype_compat.0 {
                                    Ok(col_ix)
                                } else {
                                    Err(InsertDataError::DatumIncompatibleWithColumn{
                                        col: col.to_owned(),
                                        ftype: ftype_compat.1.ftype,
                                        ftype_req: ftype_compat.1.ftype_req,
                                    })
                                }
                            }
                            // if this is a new column
                            None => {
                                new_cols.insert(col.to_owned());
                                col_ix_from_lookup(col, &ix_lookup)
                                    .map(|ix| ix + ncols)
                            }
                        }
                        .map(|col_ix| {
                            index_row.values.push(IndexValue {
                                col_ix,
                                value: value.value.clone(),
                            });
                        })
                    })
                    .map(|_| index_rows.push(index_row))
            }
        })
        .map(|_| {
            (new_rows, new_cols, overwrite_missing, overwrite_present)
        })
    }?;

    let tasks = InsertDataTasks {
        new_rows,
        new_cols,
        overwrite_missing,
        overwrite_present,
    };

    Ok((tasks, index_rows))
}

pub(crate) fn add_categories(
    rows: &[Row],
    mut engine: &mut Engine,
    mode: WriteMode,
) -> Result<HashMap<usize, (usize, usize)>, InsertDataError> {
    // lookup by index, get (k_before, k_after)
    let mut cat_lookup: HashMap<usize, (usize, usize)> = HashMap::new();
    rows.iter().try_for_each(|row| {
        row.values.iter().try_for_each(|value| {
            let col_name = &value.col_name;
            engine
                .codebook
                .col_metadata
                .get(col_name)
                .map(|(ix, colmd)| match colmd.coltype {
                    // Get the number of categories, k.
                    ColType::Categorical { k, .. } => {
                        match value.value {
                            Datum::Categorical(x) => Ok(Some(x)),
                            Datum::Missing => Ok(None),
                            _ => Err(
                                InsertDataError::DatumIncompatibleWithColumn {
                                    col: col_name.into(),
                                    ftype_req: FType::Categorical,
                                    // this should never fail because TryFrom only
                                    // fails for Datum::Missing, and that case is
                                    // handled above
                                    ftype: (&value.value).try_into().unwrap(),
                                },
                            ),
                        }
                        .map(|value| {
                            match value {
                                Some(x) => {
                                    let (_, ncats_req) =
                                        cat_lookup.entry(ix).or_insert((k, k));
                                    if x as usize >= *ncats_req {
                                        // use x + 1 because x is an index and
                                        // ncats is a length.
                                        *ncats_req = x as usize + 1;
                                    };
                                }
                                None => (),
                            }
                        })
                    }
                    _ => Ok(()),
                })
                .unwrap_or(Ok(())) // NoneError means column is new. Ignore.
        })
    })?;

    let mut cols_extended: HashMap<usize, (usize, usize)> = HashMap::new();

    for (ix, (ncats, ncats_req)) in cat_lookup.drain() {
        if ncats_req > ncats {
            if mode.allow_extend_support {
                // we want more categories than we have, and the user has
                // allowed support extension
                increase_column_categories(&mut engine, ix, ncats_req);
                // add
                cols_extended.insert(ix, (ncats, ncats_req));
            } else {
                // support extension not allowed
                return Err(InsertDataError::ModeForbidsCategoryExtension);
            }
        }
    }

    Ok(cols_extended)
}

fn increase_column_categories(
    engine: &mut Engine,
    col_ix: usize,
    ncats_req: usize,
) {
    // Adjust in codebook
    // FIXME: how to update value map?
    match engine.codebook.col_metadata[col_ix].coltype {
        ColType::Categorical { ref mut k, .. } => *k = ncats_req,
        _ => panic!("Tried to change cardinality of non-categorical column"),
    }
    // Adjust component models, priors, suffstats
    engine
        .states
        .iter_mut()
        .for_each(|state| match state.feature_mut(col_ix) {
            ColModel::Categorical(column) => {
                column.prior.symdir = SymmetricDirichlet::new_unchecked(
                    column.prior.symdir.alpha(),
                    ncats_req,
                );
                column.components.iter_mut().for_each(|cpnt| {
                    cpnt.stat = CategoricalSuffStat::from_parts_unchecked(
                        cpnt.stat.n(),
                        {
                            let mut counts = cpnt.stat.counts().to_owned();
                            counts.resize(ncats_req, 0.0);
                            counts
                        },
                    );

                    cpnt.fx = Categorical::new_unchecked({
                        let mut ln_weights = cpnt.fx.ln_weights().to_owned();
                        ln_weights.resize(ncats_req, NEG_INFINITY);
                        ln_weights
                    });
                })
            }
            _ => panic!("Requested non-categorical column"),
        })
}

pub(crate) fn create_new_columns<R: rand::Rng>(
    col_metadata: &ColMetadataList,
    state_shape: (usize, usize),
    mut rng: &mut R,
) -> Result<Vec<ColModel>, InsertDataError> {
    let (nrows, ncols) = state_shape;
    col_metadata
        .iter()
        .enumerate()
        .map(|(i, colmd)| match &colmd.coltype {
            ColType::Continuous { hyper } => {
                let data: SparseContainer<f64> =
                    SparseContainer::all_missing(nrows);
                if let Some(h) = hyper {
                    let id = i + ncols;
                    let prior = Ng::from_hyper(h.clone(), &mut rng);
                    let column = Column::new(id, data, prior);
                    Ok(ColModel::Continuous(column))
                } else {
                    Err(InsertDataError::NoGaussianHyperForNewColumn(
                        colmd.name.clone(),
                    ))
                }
            }
            ColType::Count { hyper } => {
                let data: SparseContainer<u32> =
                    SparseContainer::all_missing(nrows);
                if let Some(h) = hyper {
                    let id = i + ncols;
                    let prior = Pg::from_hyper(h.clone(), &mut rng);
                    let column = Column::new(id, data, prior);
                    Ok(ColModel::Count(column))
                } else {
                    Err(InsertDataError::NoPoissonHyperForNewColumn(
                        colmd.name.clone(),
                    ))
                }
            }
            ColType::Categorical { k, hyper, .. } => {
                let data: SparseContainer<u8> =
                    SparseContainer::all_missing(nrows);

                let prior = hyper
                    .as_ref()
                    .map(|h| Csd::from_hyper(*k, h.clone(), &mut rng))
                    .unwrap_or_else(|| Csd::vague(*k, &mut rng));

                let id = i + ncols;
                let column = Column::new(id, data, prior);
                Ok(ColModel::Categorical(column))
            }
            ColType::Labeler {
                n_labels,
                pr_h,
                pr_k,
                pr_world,
            } => {
                let data: SparseContainer<Label> =
                    SparseContainer::all_missing(nrows);
                let default_prior = LabelerPrior::standard(*n_labels);
                let prior = LabelerPrior {
                    pr_h: pr_h
                        .as_ref()
                        .map_or(default_prior.pr_h, |p| p.to_owned()),
                    pr_k: pr_k
                        .as_ref()
                        .map_or(default_prior.pr_k, |p| p.to_owned()),
                    pr_world: pr_world
                        .as_ref()
                        .map_or(default_prior.pr_world, |p| p.to_owned()),
                };
                let id = i + ncols;
                let column = Column::new(id, data, prior);
                Ok(ColModel::Labeler(column))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::Example;
    use braid_codebook::{ColMetadata, ColType};

    #[test]
    fn errors_when_no_col_metadata_when_new_columns() {
        let engine = Example::Animals.engine().unwrap();
        let moose_updates = Row {
            row_name: "moose".into(),
            values: vec![
                Value {
                    col_name: "does+taxes".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };

        let result = insert_data_tasks(&vec![moose_updates], &None, &engine);

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
        let moose_updates = Row {
            row_name: "moose".into(),
            values: vec![
                Value {
                    col_name: "does+taxes".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };

        let col_metadata = ColMetadataList::new(vec![ColMetadata {
            name: "dances".into(),
            coltype: ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        let result = insert_data_tasks(
            &vec![moose_updates],
            &Some(col_metadata),
            &engine,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::NewColumnNotInColumnMetadata("does+taxes".into())
        );
    }

    #[test]
    fn tasks_on_one_existing_row() {
        let engine = Example::Animals.engine().unwrap();
        let moose_updates = Row {
            row_name: "moose".into(),
            values: vec![
                Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };
        let (tasks, ixrows) =
            insert_data_tasks(&vec![moose_updates], &None, &engine).unwrap();

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
                        value: Datum::Categorical(1)
                    },
                    IndexValue {
                        col_ix: 34,
                        value: Datum::Categorical(1)
                    },
                ]
            }]
        );
    }

    #[test]
    fn tasks_on_one_new_row() {
        let engine = Example::Animals.engine().unwrap();
        let pegasus = Row {
            row_name: "pegasus".into(),
            values: vec![
                Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };
        let (tasks, ixrows) =
            insert_data_tasks(&vec![pegasus], &None, &engine).unwrap();

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
                        value: Datum::Categorical(1)
                    },
                    IndexValue {
                        col_ix: 34,
                        value: Datum::Categorical(1)
                    },
                ]
            }]
        );
    }

    #[test]
    fn tasks_on_two_new_rows() {
        let engine = Example::Animals.engine().unwrap();
        let pegasus = Row {
            row_name: "pegasus".into(),
            values: vec![
                Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };

        let man = Row {
            row_name: "man".into(),
            values: vec![
                Value {
                    col_name: "smart".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "hunter".into(),
                    value: Datum::Categorical(0),
                },
            ],
        };
        let (tasks, ixrows) =
            insert_data_tasks(&vec![pegasus, man], &None, &engine).unwrap();

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
                            value: Datum::Categorical(1)
                        },
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(1)
                        },
                    ]
                },
                IndexRow {
                    row_ix: 51,
                    values: vec![
                        IndexValue {
                            col_ix: 80,
                            value: Datum::Categorical(1)
                        },
                        IndexValue {
                            col_ix: 58,
                            value: Datum::Categorical(0)
                        },
                    ]
                }
            ]
        );
    }

    #[test]
    fn tasks_on_one_new_and_one_existin_row() {
        let engine = Example::Animals.engine().unwrap();
        let pegasus = Row {
            row_name: "pegasus".into(),
            values: vec![
                Value {
                    col_name: "swims".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };

        let moose = Row {
            row_name: "moose".into(),
            values: vec![
                Value {
                    col_name: "smart".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "hunter".into(),
                    value: Datum::Categorical(0),
                },
            ],
        };
        let (tasks, ixrows) =
            insert_data_tasks(&vec![pegasus, moose], &None, &engine).unwrap();

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
                            value: Datum::Categorical(1)
                        },
                        IndexValue {
                            col_ix: 34,
                            value: Datum::Categorical(1)
                        },
                    ]
                },
                IndexRow {
                    row_ix: 15,
                    values: vec![
                        IndexValue {
                            col_ix: 80,
                            value: Datum::Categorical(1)
                        },
                        IndexValue {
                            col_ix: 58,
                            value: Datum::Categorical(0)
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
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();
        let moose_updates = Row {
            row_name: "moose".into(),
            values: vec![
                Value {
                    col_name: "dances".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };
        let (tasks, ixrows) = insert_data_tasks(
            &vec![moose_updates],
            &Some(col_metadata),
            &engine,
        )
        .unwrap();

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
                        value: Datum::Categorical(1)
                    },
                    IndexValue {
                        col_ix: 34,
                        value: Datum::Categorical(1)
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
                value_map: None,
            },
            notes: None,
        }])
        .unwrap();

        let peanut = Row {
            row_name: "peanut".into(),
            values: vec![
                Value {
                    col_name: "dances".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(0),
                },
            ],
        };
        let (tasks, ixrows) =
            insert_data_tasks(&vec![peanut], &Some(col_metadata), &engine)
                .unwrap();

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
                        value: Datum::Categorical(1)
                    },
                    IndexValue {
                        col_ix: 34,
                        value: Datum::Categorical(0)
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
                    value_map: None,
                },
                notes: None,
            },
            ColMetadata {
                name: "eats+figs".into(),
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    value_map: None,
                },
                notes: None,
            },
        ])
        .unwrap();

        let moose_updates = Row {
            row_name: "moose".into(),
            values: vec![
                Value {
                    col_name: "flys".into(),
                    value: Datum::Categorical(1),
                },
                Value {
                    col_name: "eats+figs".into(),
                    value: Datum::Categorical(0),
                },
                Value {
                    col_name: "dances".into(),
                    value: Datum::Categorical(1),
                },
            ],
        };
        let (tasks, ixrows) = insert_data_tasks(
            &vec![moose_updates],
            &Some(col_metadata),
            &engine,
        )
        .unwrap();

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
                        value: Datum::Categorical(1)
                    },
                    IndexValue {
                        col_ix: 86,
                        value: Datum::Categorical(0)
                    },
                    IndexValue {
                        col_ix: 85,
                        value: Datum::Categorical(1)
                    },
                ]
            }]
        );
    }
}

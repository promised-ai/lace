use super::error::InsertDataError;
use crate::{Engine, OracleT};
use braid_stats::Datum;
use indexmap::IndexSet;
use std::collections::HashSet;

/// Defines the overwrite behavior of insert datum
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InsertOverwrite {
    /// Overwrite anything
    Allow,
    /// Do not overwrite any existing cells. Only allow data in new rows or
    /// columns.
    Deny,
    /// Same as deny, but also allow existing cells that are empty to be
    /// overwritten.
    MissingOnly,
}

/// Defines insert data behavior
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InsertMode {
    /// Can add new rows or column
    Unrestricted(InsertOverwrite),
    /// Cannot add new rows, but can add new columns
    DenyNewRows(InsertOverwrite),
    /// Cannot add new columns, but can add new rows
    DenyNewColumns(InsertOverwrite),
    /// No adding new rows or columns
    DenyNewRowsAndColumns(InsertOverwrite),
}

impl InsertMode {
    /// Retrieve overwrite behavior
    pub fn overwrite(&self) -> InsertOverwrite {
        match self {
            Self::Unrestricted(overwrite) => *overwrite,
            Self::DenyNewRows(overwrite) => *overwrite,
            Self::DenyNewColumns(overwrite) => *overwrite,
            Self::DenyNewRowsAndColumns(overwrite) => *overwrite,
        }
    }
}

/// A datum for insertion into a certain column
pub struct Value {
    /// Name of the column
    pub col_name: String,
    /// The value of the cell
    pub value: Datum,
}

/// A list of data for insertion into a certain row
///
/// ``` ignore
/// use braid::examples::Example;
/// use braid::examples::animals::{Column, Row};
/// use braid::engine::{InsertOverwrite, InsertMode};
/// use braid::OracleT;
///
/// let engine = Example::Animals.engine().unwrap();
///
/// let pegasus = Row {
///     row_name: "pegasus".to_string(),
///     data: vec![
///         Value {
///             col_name: "hooves".to_string(),
///             value: Datum::Categorical(1),
///         },
///         Value {
///             col_name: "flys".to_string(),
///             value: Datum::Categorical(1),
///         },
///         Value {
///             col_name: "furry".to_string(),
///             value: Datum::Categorical(1),
///         }
///     ]
/// }
///
/// let nrows = engine.nrows();
/// let ncols = engine.cols();
///
/// // Can only insert data into a new row
/// let insert_mode = InsertMode::DenyNewColumns(InsertOverwrite::Deny);
/// let result = engine.insert_data(vec![pegasus], None, insert_mode);
///
/// // Added a row
/// assert!(result.is_ok());
/// assert_eq!(engine.ncols(), ncols);
///
/// // The new entries are in.
/// assert_eq!(engine.datum(nrows, Column::hooves), Datum::Categorical(1));
/// assert_eq!(engine.datum(nrows, Column::flys), Datum::Categorical(1));
/// assert_eq!(engine.datum(nrows, Column::furry), Datum::Categorical(1));
///
/// // Other entries in the row are missing
/// assert_eq!(engine.datum(nrows, Column::swims), Datum::Missing);
/// ```
pub struct Row {
    /// The name of the row
    pub row_name: String,
    /// The cells and values to fill in
    pub values: Vec<Value>,
}

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

use crate::cc::ColModel;
use braid_codebook::Codebook;
use braid_codebook::ColType;

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
        mode: &InsertMode,
    ) -> Result<(), InsertDataError> {
        match mode {
            InsertMode::DenyNewRows(_) => {
                if !self.new_rows.is_empty() {
                    Err(InsertDataError::ModeForbidsNewRowsError)
                } else {
                    Ok(())
                }
            }
            InsertMode::DenyNewColumns(_) => {
                if !self.new_cols.is_empty() {
                    Err(InsertDataError::ModeForbidsNewColumnsError)
                } else {
                    Ok(())
                }
            }
            InsertMode::DenyNewRowsAndColumns(_) => {
                if !(self.new_rows.is_empty() && self.new_cols.is_empty()) {
                    Err(InsertDataError::ModeForbidsNewRowsOrColumnsError)
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }
}

use braid_utils::ForEachOk;
use std::collections::HashMap;

fn ix_lookup_from_cdebook<'a>(
    partial_codebook: &'a Option<Codebook>,
) -> Option<HashMap<&'a str, usize>> {
    partial_codebook.as_ref().map(|cb| {
        cb.col_metadata
            .iter()
            .enumerate()
            .map(|(ix, md)| (md.name.as_str(), ix))
            .collect()
    })
}

fn col_ix_from_lookup(
    col: &str,
    lookup: &Option<HashMap<&str, usize>>,
) -> Result<usize, InsertDataError> {
    match lookup {
        Some(lkp) => lkp
            .get(col)
            .ok_or_else(|| {
                println!("{} errored", col);
                InsertDataError::NewColumnNotInPartialCodebookError(
                    col.to_owned(),
                )
            })
            .map(|col| *col),
        None => Err(InsertDataError::NoPartialCodebookError),
    }
}

/// Get a summary of the tasks required to insert `rows` into `Engine`.
pub(crate) fn insert_data_tasks(
    rows: &Vec<Row>,
    partial_codebook: &Option<Codebook>,
    engine: &Engine,
) -> Result<(InsertDataTasks, Vec<IndexRow>), InsertDataError> {
    // Get a map into the new column indices if they exist
    let ix_lookup = ix_lookup_from_cdebook(&partial_codebook);

    // Get a list of all the row names. The row names must be included in the
    // codebook in order to insert data.
    let row_names: &Vec<_> = match engine.codebook.row_names {
        Some(ref row_names) => Ok(row_names),
        None => Err(InsertDataError::NoRowNamesInCodebookError),
    }?;

    let mut index_rows: Vec<IndexRow> = Vec::new();
    let mut nrows = engine.nrows();
    let ncols = engine.ncols();

    let (new_rows, new_cols, overwrite_missing, overwrite_present) = {
        let mut overwrite_missing = false;
        let mut overwrite_present = false;
        let mut new_rows: IndexSet<String> = IndexSet::new();
        let mut new_cols: HashSet<String> = HashSet::new();
        rows.iter().for_each_ok(|row| {
            if !new_rows.contains(&row.row_name)
                && !row_names.iter().any(|name| name == &row.row_name)
            {
                // If the row does not exist..
                let mut index_row = IndexRow {
                    row_ix: nrows,
                    values: vec![],
                };
                nrows += 1;
                new_rows.insert(row.row_name.clone());
                row.values
                    .iter()
                    .for_each_ok(|value| {
                        let col = &value.col_name;
                        let colmd = engine.codebook.col_metadata.get(col);

                        if !new_cols.contains(col) && colmd.is_none() {
                            new_cols.insert(col.to_owned());
                        }

                        match colmd {
                            Some((col_ix, _)) => {
                                if engine.ftype(col_ix).unwrap().datum_compatible(&value.value) {
                                    Ok(col_ix)
                                } else {
                                    Err(InsertDataError::DatumIncompatibleWithColumn(col.to_owned()))
                                }
                            },
                            None => col_ix_from_lookup(col, &ix_lookup)
                                .map(|ix| ix + ncols),
                        }
                        .map(|col_ix| {
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
                let (row_ix, _) = row_names
                    .iter()
                    .enumerate()
                    .find(|(_, name)| name == &&row.row_name)
                    .expect("Unable to get row index");

                let mut index_row = IndexRow {
                    row_ix,
                    values: vec![],
                };

                row.values
                    .iter()
                    .for_each_ok(|value| {
                        let col = &value.col_name;
                        let colmd = engine.codebook.col_metadata.get(col);

                        match colmd {
                            // if this is a existing cell
                            Some(md) => {
                                let col_ix = engine
                                    .codebook
                                    .col_metadata
                                    .get(&col)
                                    .unwrap()
                                    .0;

                                if engine
                                    .datum(row_ix, col_ix)
                                    .unwrap()
                                    .is_missing()
                                {
                                    overwrite_missing = true;
                                } else {
                                    overwrite_present = true;
                                }

                                if engine.ftype(col_ix).unwrap().datum_compatible(&value.value) {
                                    Ok(col_ix)
                                } else {
                                    Err(InsertDataError::DatumIncompatibleWithColumn(col.to_owned()))
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

use crate::cc::Column;
use crate::cc::DataContainer;
use braid_stats::labeler::{Label, LabelerPrior};
use braid_stats::prior::{Csd, Ng};

pub(crate) fn create_new_columns<R: rand::Rng>(
    partial_codebook: &Codebook,
    state_shape: (usize, usize),
    new_columns: &HashSet<String>,
    mut rng: &mut R,
) -> Result<Vec<ColModel>, InsertDataError> {
    let (nrows, ncols) = state_shape;
    partial_codebook
        .col_metadata
        .iter()
        .enumerate()
        .map(|(i, colmd)| match &colmd.coltype {
            ColType::Continuous { hyper } => {
                let data: DataContainer<f64> =
                    DataContainer::all_missing(nrows);
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
            ColType::Categorical { k, hyper, .. } => {
                let data: DataContainer<u8> = DataContainer::all_missing(nrows);

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
                let data: DataContainer<Label> =
                    DataContainer::all_missing(nrows);
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
    use braid_codebook::{ColMetadata, ColType, SpecType};
    use std::convert::TryInto;

    #[test]
    fn errors_when_no_partial_codebook_when_no_columns() {
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
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::NoPartialCodebookError
        );
    }

    #[test]
    fn errors_when_new_column_not_in_partial_codebook() {
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

        let partial_codebook = Codebook {
            table_name: "updates".into(),
            state_alpha_prior: None,
            view_alpha_prior: None,
            col_metadata: vec![ColMetadata {
                name: "dances".into(),
                spec_type: SpecType::Other,
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    value_map: None,
                },
                notes: None,
            }]
            .try_into()
            .unwrap(),
            comments: None,
            row_names: None,
        };

        let result = insert_data_tasks(
            &vec![moose_updates],
            &Some(partial_codebook),
            &engine,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            InsertDataError::NewColumnNotInPartialCodebookError(
                "does+taxes".into()
            )
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
        let partial_codebook = Codebook {
            table_name: "updates".into(),
            state_alpha_prior: None,
            view_alpha_prior: None,
            col_metadata: vec![ColMetadata {
                name: "dances".into(),
                spec_type: SpecType::Other,
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    value_map: None,
                },
                notes: None,
            }]
            .try_into()
            .unwrap(),
            comments: None,
            row_names: None,
        };
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
            &Some(partial_codebook),
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
        let partial_codebook = Codebook {
            table_name: "updates".into(),
            state_alpha_prior: None,
            view_alpha_prior: None,
            col_metadata: vec![ColMetadata {
                name: "dances".into(),
                spec_type: SpecType::Other,
                coltype: ColType::Categorical {
                    k: 2,
                    hyper: None,
                    value_map: None,
                },
                notes: None,
            }]
            .try_into()
            .unwrap(),
            comments: None,
            row_names: None,
        };
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
            insert_data_tasks(&vec![peanut], &Some(partial_codebook), &engine)
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
        let partial_codebook = Codebook {
            table_name: "updates".into(),
            state_alpha_prior: None,
            view_alpha_prior: None,
            col_metadata: vec![
                ColMetadata {
                    name: "dances".into(),
                    spec_type: SpecType::Other,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        value_map: None,
                    },
                    notes: None,
                },
                ColMetadata {
                    name: "eats+figs".into(),
                    spec_type: SpecType::Other,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        value_map: None,
                    },
                    notes: None,
                },
            ]
            .try_into()
            .unwrap(),
            comments: None,
            row_names: None,
        };
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
            &Some(partial_codebook),
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

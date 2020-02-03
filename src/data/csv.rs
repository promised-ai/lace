//! Utilities to process data or derive a `Codebook` from a CSV.
//!
//! # CSV data requirements
//!
//! - The first row of the CSV must have a header
//! - The first column of the csv must be `ID`
//! - All columns in the csv, other than `ID`, must be in the codebook
//! - Missing data are empty cells
//!
//! ## Categorical Data requirements
//!
//! - Categorical data *input* must be integers
//! - The minimum value must be 0, and the maximum value must be k - 1
//!
//! **NOTE:** Codebooks generated from a CSV will create value maps that map
//! integer indices to string values, but the data used to create an `Engine`
//! must contain only integers. Use the value map to convert the strings in
//! your CSV to integers.
//!
//! If you have a CSV like this
//!
//! ```text
//! ID,x,y
//! 0,1,dog
//! 1,1,cat
//! 2,2,cat
//! ```
//!
//! Your value map may look like this
//!
//! ```json
//! {
//!     0: "dog",
//!     1: "cat"
//! }
//! ```
//!
//! You would then use the value map to make the input CSV look like this
//!
//! ```text
//! ID,x,y
//! 0,1,0
//! 1,1,1
//! 2,2,1
//! ```
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::{f64, io::Read};

use braid_codebook::{Codebook, ColMetadata, ColType};
use braid_stats::labeler::{Label, LabelerPrior};
use braid_stats::prior::{Csd, Ng, NigHyper};
use braid_stats::Datum;
use braid_utils::{parse_result, ForEachOk};
use csv::{Reader, StringRecord};
use rv::dist::{Categorical, Gaussian};

use super::error::csv::CsvParseError;
use crate::cc::{
    AppendRowsData, ColModel, Column, DataContainer, FType, Feature,
};

/// Convert a csv with a codebook into data to new row data to append to a
/// state
pub fn row_data_from_csv<R: Read>(
    mut reader: Reader<R>,
    mut codebook: &mut Codebook,
) -> Result<Vec<AppendRowsData>, CsvParseError> {
    // We need to sort the column metadatas into the same order as the
    // columns appear in the csv file.
    let colmds = {
        // headers() borrows mutably from the reader, so it has to go in its
        // own scope.
        let csv_header = reader.headers().unwrap();
        colmds_by_header(&codebook, &csv_header)
    }?;

    let mut row_data: Vec<AppendRowsData> = colmds
        .iter()
        .map(|(id, _)| AppendRowsData::new(*id, Vec::new()))
        .collect();

    for record in reader.records() {
        row_data = push_row_to_row_data(
            &mut codebook,
            &colmds,
            row_data,
            record.unwrap(),
        )?;
    }

    Ok(row_data)
}

fn push_row_to_row_data(
    codebook: &mut Codebook,
    colmds: &[(usize, ColMetadata)],
    mut row_data: Vec<AppendRowsData>,
    record: StringRecord,
) -> Result<Vec<AppendRowsData>, CsvParseError> {
    let mut record_iter = record.iter();
    let row_name: String = record_iter
        .next()
        .ok_or(CsvParseError::NoColumnsError)?
        .to_owned();

    codebook
        .row_names
        .insert(row_name.clone())
        .map_err(|_| CsvParseError::DuplicateCsvRowsError)?;

    colmds.iter().zip(record.iter().skip(1)).for_each_ok(
        |((id, colmd), rec)| {
            match colmd.coltype {
                ColType::Continuous { .. } => parse_result::<f64>(rec)
                    .map_err(|_| CsvParseError::InvalidValueForColumnError {
                        col_id: *id,
                        row_name: row_name.clone(),
                        val: String::from(rec),
                        col_type: FType::Continuous,
                    })
                    .map(|res| {
                        res.map_or_else(|| Datum::Missing, Datum::Continuous)
                    }),
                ColType::Categorical { .. } => parse_result::<u8>(rec)
                    .map_err(|_| CsvParseError::InvalidValueForColumnError {
                        col_id: *id,
                        row_name: row_name.clone(),
                        val: String::from(rec),
                        col_type: FType::Categorical,
                    })
                    .map(|res| {
                        res.map_or_else(|| Datum::Missing, Datum::Categorical)
                    }),
                ColType::Labeler { .. } => parse_result::<Label>(rec)
                    .map_err(|_| CsvParseError::InvalidValueForColumnError {
                        col_id: *id,
                        row_name: row_name.clone(),
                        val: String::from(rec),
                        col_type: FType::Labeler,
                    })
                    .map(|res| {
                        res.map_or_else(|| Datum::Missing, Datum::Label)
                    }),
            }
            .map(|datum| {
                // This is our error, not a user error
                if row_data[*id].col_ix != *id {
                    panic!(
                        "row data index ({}) does not match column metadata ID
                        ({})",
                        row_data[*id].col_ix, id
                    );
                };
                row_data[*id].data.push(datum);
            })
        },
    )?;

    Ok(row_data)
}

fn get_continuous_prior<R: rand::Rng>(
    ftr: &Column<f64, Gaussian, Ng>,
    codebook: &Codebook,
    mut rng: &mut R,
) -> Ng {
    match &codebook.col_metadata[ftr.id].coltype {
        ColType::Continuous { hyper: Some(h) } => {
            Ng::from_hyper(h.to_owned(), &mut rng)
        }
        ColType::Continuous { hyper: None } => {
            Ng::from_data(&ftr.data.data, &mut rng)
        }
        _ => panic!("expected ColType::Continuous for column {}", ftr.id()),
    }
}

fn get_categorical_prior<R: rand::Rng>(
    ftr: &Column<u8, Categorical, Csd>,
    codebook: &Codebook,
    mut rng: &mut R,
) -> Csd {
    match &codebook.col_metadata[ftr.id].coltype {
        ColType::Categorical {
            k, hyper: Some(h), ..
        } => Csd::from_hyper(*k, h.to_owned(), &mut rng),
        ColType::Categorical { k, hyper: None, .. } => Csd::vague(*k, &mut rng),
        _ => panic!("expected ColType::Categorical for column {}", ftr.id()),
    }
}

/// Reads the columns of a csv into a vector of `ColModel`.
pub fn read_cols<R: Read>(
    mut reader: Reader<R>,
    codebook: &Codebook,
) -> Result<Vec<ColModel>, CsvParseError> {
    // TODO: pass in Rng for seed control
    let mut rng = rand::thread_rng();
    // We need to sort the column metadatas into the same order as the
    // columns appear in the csv file.
    let colmds = {
        // headers() borrows mutably from the reader, so it has to go in its
        // own scope.
        let csv_header = reader.headers().unwrap();
        colmds_by_header(&codebook, &csv_header)
    }?;

    let lookups: Vec<Option<HashMap<String, usize>>> = colmds
        .iter()
        .map(|(_, colmd)| colmd.coltype.lookup())
        .collect();

    let mut col_models = reader.records().fold(
        Ok(init_col_models(&colmds)),
        |col_models_res, record| {
            if let Ok(col_models) = col_models_res {
                push_row_to_col_models(col_models, record.unwrap(), &lookups)
            } else {
                col_models_res
            }
        },
    )?;

    if col_models
        .iter()
        .any(|cm| cm.len() != codebook.row_names.len())
    {
        return Err(CsvParseError::CodebookAndDataRowMismatchErr);
    }

    col_models.iter_mut().for_each(|col_model| match col_model {
        ColModel::Continuous(ftr) => {
            let prior = get_continuous_prior(&ftr, &codebook, &mut rng);
            ftr.prior = prior;
        }
        ColModel::Categorical(ftr) => {
            let prior = get_categorical_prior(&ftr, &codebook, &mut rng);
            ftr.prior = prior;
        }
        // Labeler type priors are injected from the codebook or from
        // LabelerPrior::standard
        _ => (),
    });
    Ok(col_models)
}

fn push_row_to_col_models(
    mut col_models: Vec<ColModel>,
    record: StringRecord,
    lookups: &[Option<HashMap<String, usize>>],
) -> Result<Vec<ColModel>, CsvParseError> {
    let mut record_iter = record.iter();
    let row_name: String = record_iter
        .next()
        .ok_or(CsvParseError::NoColumnsError)?
        .into();

    col_models
        .iter_mut()
        .zip(record_iter) // assume id is the first column
        .zip(lookups)
        .for_each_ok(|((cm, rec), lookup_opt)| {
            match cm {
                ColModel::Continuous(ftr) => {
                    // TODO: Check for NaN, -Inf, and Inf
                    parse_result::<f64>(rec)
                        .map_err(|_| {
                            CsvParseError::InvalidValueForColumnError {
                                col_id: ftr.id(),
                                row_name: row_name.clone(),
                                val: String::from(rec),
                                col_type: ftr.ftype(),
                            }
                        })
                        .map(|val_opt| ftr.data.push(val_opt))
                }
                ColModel::Categorical(ftr) => {
                    // check if empty cell
                    if rec.trim() == "" {
                        ftr.data.push(None);
                        Ok(())
                    } else if let Some(lookup) = lookup_opt {
                        lookup
                            .get(&rec.to_string())
                            .and_then(|&value| u8::try_from(value).ok())
                            .ok_or(CsvParseError::InvalidValueForColumnError {
                                col_id: ftr.id(),
                                row_name: row_name.clone(),
                                val: String::from(rec),
                                col_type: ftr.ftype(),
                            })
                            .map(|val| ftr.data.push(Some(val)))
                    } else {
                        parse_result::<u8>(rec)
                            .map_err(|_| {
                                CsvParseError::InvalidValueForColumnError {
                                    col_id: ftr.id(),
                                    row_name: row_name.clone(),
                                    val: String::from(rec),
                                    col_type: ftr.ftype(),
                                }
                            })
                            .map(|val_opt| ftr.data.push(val_opt))
                    }
                }
                ColModel::Labeler(ftr) => parse_result::<Label>(rec)
                    .map_err(|_| CsvParseError::InvalidValueForColumnError {
                        col_id: ftr.id(),
                        row_name: row_name.clone(),
                        val: String::from(rec),
                        col_type: ftr.ftype(),
                    })
                    .map(|val_opt| ftr.data.push(val_opt)),
            }
        })?;

    Ok(col_models)
}

pub fn init_col_models(colmds: &[(usize, ColMetadata)]) -> Vec<ColModel> {
    let mut rng = rand::thread_rng();
    colmds
        .iter()
        .map(|(id, colmd)| {
            match colmd.coltype {
                // Ignore hypers until all the data are loaded, then we'll
                // re-initialize
                ColType::Continuous { .. } => {
                    let data = DataContainer::new(vec![]);
                    let prior = {
                        let h = NigHyper::default();
                        Ng::from_hyper(h, &mut rng)
                    };
                    let column = Column::new(*id, data, prior);
                    ColModel::Continuous(column)
                }
                ColType::Categorical { k, .. } => {
                    let data = DataContainer::new(vec![]);
                    let prior = { Csd::vague(k, &mut rng) };
                    let column = Column::new(*id, data, prior);
                    ColModel::Categorical(column)
                }
                ColType::Labeler {
                    n_labels,
                    ref pr_h,
                    ref pr_k,
                    ref pr_world,
                } => {
                    let data = DataContainer::new(vec![]);
                    let default_prior = LabelerPrior::standard(n_labels);
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
                    let column = Column::new(*id, data, prior);
                    ColModel::Labeler(column)
                }
            }
        })
        .collect()
}

// get column (id, metadata) from a codebook only for the columns in the header
fn colmds_by_header(
    codebook: &Codebook,
    csv_header: &StringRecord,
) -> Result<Vec<(usize, ColMetadata)>, CsvParseError> {
    let colmds = &codebook.col_metadata;
    let mut csv_columns: HashSet<String> = HashSet::new();
    let mut header_iter = csv_header.iter();

    header_iter
        .next()
        .ok_or(CsvParseError::NoColumnsError)
        .and_then(|col_name| {
            if col_name.to_lowercase() != "id" {
                Err(CsvParseError::FirstColumnNotNamedIdError)
            } else {
                Ok(())
            }
        })?;

    let output: Result<Vec<(usize, ColMetadata)>, CsvParseError> = header_iter
        .map(|col_name| {
            let col = String::from(col_name);
            colmds
                .get(&col)
                .ok_or(CsvParseError::MissingCodebookColumnsError)
                .and_then(|(id, colmd)| {
                    if col != colmd.name {
                        Err(CsvParseError::CsvCodebookColumnsMisorderedError)
                    } else if csv_columns.insert(col.clone()) {
                        Ok((id, colmd.clone()))
                    } else {
                        Err(CsvParseError::DuplicateCsvColumnsError)
                    }
                })
        })
        .collect();

    output.and_then(|out| {
        if colmds.len() != out.len() {
            Err(CsvParseError::MissingCsvColumnsError)
        } else {
            Ok(out)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::Feature;
    use approx::*;
    use braid_codebook::{ColMetadata, ColMetadataList, RowNameList, SpecType};
    use csv::ReaderBuilder;
    use indoc::indoc;
    use maplit::btreemap;

    fn get_codebook(n_rows: usize) -> Codebook {
        Codebook {
            view_alpha_prior: None,
            state_alpha_prior: None,
            comments: None,
            table_name: String::from("test"),
            row_names: RowNameList::from_range(0..n_rows),
            col_metadata: ColMetadataList::new(vec![
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: String::from("x"),
                    coltype: ColType::Continuous { hyper: None },
                    notes: None,
                },
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: String::from("y"),
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                    notes: None,
                },
            ])
            .unwrap(),
        }
    }

    fn get_codebook_value_map(n_rows: usize) -> Codebook {
        Codebook {
            view_alpha_prior: None,
            state_alpha_prior: None,
            comments: None,
            table_name: String::from("test"),
            row_names: RowNameList::from_range(0..n_rows),
            col_metadata: ColMetadataList::new(vec![
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: String::from("x"),
                    coltype: ColType::Continuous { hyper: None },
                    notes: None,
                },
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: String::from("y"),
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: Some(btreemap! {
                           0 => String::from("dog"),
                           1 => String::from("cat"),
                        }),
                    },
                    notes: None,
                },
            ])
            .unwrap(),
        }
    }

    fn data_with_no_missing() -> (String, Codebook) {
        let data = "ID,x,y\n0,0.1,0\n1,1.2,0\n2,2.3,1";
        // NOTE that the metadatas and the csv column names are in different
        // order
        (String::from(data), get_codebook(3))
    }

    fn data_with_some_missing() -> (String, Codebook) {
        let data = "ID,x,y\n0,,0\n1,1.2,0\n2,2.3,";
        // NOTE that the metadatas and the csv column names are in different
        // order
        (String::from(data), get_codebook(3))
    }

    fn data_with_string_no_missing() -> (String, Codebook) {
        let data = "ID,x,y\n0,0.1,\"dog\"\n1,1.2,\"cat\"\n2,2.3,\"cat\"";
        // NOTE that the metadatas and the csv column names are in different
        // order
        (String::from(data), get_codebook_value_map(3))
    }

    fn data_with_string_missing() -> (String, Codebook) {
        let data = "ID,x,y\n0,0.1,\"dog\"\n1,1.2,\"cat\"\n2,2.3,\"\"";
        // NOTE that the metadatas and the csv column names are in different
        // order
        (String::from(data), get_codebook_value_map(3))
    }

    #[test]
    fn col_mds_by_header_should_match_header_order() {
        let (data, codebook) = data_with_no_missing();
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let csv_header = &reader.headers().unwrap();
        let colmds = colmds_by_header(&codebook, &csv_header).unwrap();

        assert_eq!(colmds[0].0, 0);
        assert_eq!(colmds[1].0, 1);

        assert!(colmds[0].1.coltype.is_continuous());
        assert!(colmds[1].1.coltype.is_categorical());
    }

    #[test]
    fn init_col_models_should_match_header_order() {
        let (data, codebook) = data_with_no_missing();
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let csv_header = &reader.headers().unwrap();
        let colmds = colmds_by_header(&codebook, &csv_header).unwrap();
        let col_models = init_col_models(&colmds);

        assert!(col_models[0].ftype().is_continuous());
        assert!(col_models[1].ftype().is_categorical());
    }

    #[test]
    fn read_cols_standard_data() {
        let (data, codebook) = data_with_no_missing();
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        assert!(col_models[0].ftype().is_continuous());
        assert!(col_models[1].ftype().is_categorical());

        let col_x = match &col_models[0] {
            &ColModel::Continuous(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_x.data.present[0]);
        assert!(col_x.data.present[1]);
        assert!(col_x.data.present[2]);

        assert_relative_eq!(col_x.data[0], 0.1, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[1], 1.2, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[2], 2.3, epsilon = 10E-10);

        let col_y = match &col_models[1] {
            &ColModel::Categorical(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_y.data.present[0]);
        assert!(col_y.data.present[1]);
        assert!(col_y.data.present[2]);

        assert_eq!(col_y.data[0], 0);
        assert_eq!(col_y.data[1], 0);
        assert_eq!(col_y.data[2], 1);
    }

    #[test]
    fn read_cols_string_data() {
        let (data, codebook) = data_with_string_no_missing();
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        assert!(col_models[0].ftype().is_continuous());
        assert!(col_models[1].ftype().is_categorical());

        let col_x = match &col_models[0] {
            &ColModel::Continuous(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_x.data.present[0]);
        assert!(col_x.data.present[1]);
        assert!(col_x.data.present[2]);

        assert_relative_eq!(col_x.data[0], 0.1, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[1], 1.2, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[2], 2.3, epsilon = 10E-10);

        let col_y = match &col_models[1] {
            &ColModel::Categorical(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_y.data.present[0]);
        assert!(col_y.data.present[1]);
        assert!(col_y.data.present[2]);

        assert_eq!(col_y.data[0], 0);
        assert_eq!(col_y.data[1], 1);
        assert_eq!(col_y.data[2], 1);
    }

    #[test]
    fn read_cols_string_data_missing() {
        let (data, codebook) = data_with_string_missing();
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        assert!(col_models[0].ftype().is_continuous());
        assert!(col_models[1].ftype().is_categorical());

        let col_x = match &col_models[0] {
            &ColModel::Continuous(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_x.data.present[0]);
        assert!(col_x.data.present[1]);
        assert!(col_x.data.present[2]);

        assert_relative_eq!(col_x.data[0], 0.1, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[1], 1.2, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[2], 2.3, epsilon = 10E-10);

        let col_y = match &col_models[1] {
            &ColModel::Categorical(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_y.data.present[0]);
        assert!(col_y.data.present[1]);
        assert!(!col_y.data.present[2]);

        assert_eq!(col_y.data[0], 0);
        assert_eq!(col_y.data[1], 1);
        assert_eq!(col_y.data[2], u8::default());
    }

    #[test]
    fn read_cols_missing_data() {
        let (data, codebook) = data_with_some_missing();
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        assert!(col_models[0].ftype().is_continuous());
        assert!(col_models[1].ftype().is_categorical());

        let col_x = match &col_models[0] {
            &ColModel::Continuous(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(!col_x.data.present[0]);
        assert!(col_x.data.present[1]);
        assert!(col_x.data.present[2]);

        assert_relative_eq!(col_x.data[0], 0.0, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[1], 1.2, epsilon = 10E-10);
        assert_relative_eq!(col_x.data[2], 2.3, epsilon = 10E-10);

        let col_y = match &col_models[1] {
            &ColModel::Categorical(ref cm) => cm,
            _ => unreachable!(),
        };

        assert!(col_y.data.present[0]);
        assert!(col_y.data.present[1]);
        assert!(!col_y.data.present[2]);

        assert_eq!(col_y.data[0], 0);
        assert_eq!(col_y.data[1], 0);
        assert_eq!(col_y.data[2], 0);
    }

    #[test]
    fn uses_codebook_continuous_prior_if_specified() {
        let csv_data = indoc!(
            "
            id,x
            0,3.0
            1,1.1
            2,3.0
            3,1.1
            4,3.0
            5,1.6
            6,1.8"
        );

        let codebook_data = indoc!(
            "
            ---
            table_name: test
            col_metadata:
              - name: x
                coltype:
                  Continuous:
                    hyper:
                      pr_m:
                        mu: 0.0
                        sigma: 1.0
                      pr_r:
                        shape: 2.0
                        rate: 3.0
                      pr_s:
                        shape: 4.0
                        rate: 5.0
                      pr_v:
                        shape: 6.0
                        rate: 7.0
            row_names:
              - 0
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
            "
        );

        let codebook: Codebook = serde_yaml::from_str(&codebook_data).unwrap();

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        let hyper = match &col_models[0] {
            ColModel::Continuous(ftr) => ftr.prior.hyper.clone(),
            _ => panic!("wrong feature type"),
        };
        assert_relative_eq!(hyper.pr_m.mu(), 0.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_m.sigma(), 1.0, epsilon = 1E-12);

        assert_relative_eq!(hyper.pr_r.shape(), 2.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_r.rate(), 3.0, epsilon = 1E-12);

        assert_relative_eq!(hyper.pr_s.shape(), 4.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_s.rate(), 5.0, epsilon = 1E-12);

        assert_relative_eq!(hyper.pr_v.shape(), 6.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_v.rate(), 7.0, epsilon = 1E-12);
    }

    #[test]
    fn uses_codebook_categorical_prior_if_specified() {
        let csv_data = indoc!(
            "
            id,x
            0,0
            1,1
            2,0
            3,1
            4,0
            5,1
            6,1"
        );

        let codebook_data = indoc!(
            "
            ---
            table_name: test
            col_metadata:
              - name: x
                coltype:
                  Categorical:
                    k: 2 
                    hyper:
                      pr_alpha:
                        shape: 1.2
                        scale: 3.4
            row_names:
              - 0
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
            "
        );

        let codebook: Codebook = serde_yaml::from_str(&codebook_data).unwrap();

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        let hyper = match &col_models[0] {
            ColModel::Categorical(ftr) => ftr.prior.hyper.clone(),
            _ => panic!("wrong feature type"),
        };
        assert_relative_eq!(hyper.pr_alpha.shape(), 1.2, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_alpha.scale(), 3.4, epsilon = 1E-12);
    }

    // FIXME: need to make rv not worry about serializing ln_ab field in
    // kumaraswamy
    #[test]
    #[ignore]
    fn uses_codebook_labeler_prior_if_specified() {
        let csv_data = indoc!(
            r#"
            id,x
            0,"IL(0,0)"
            1,"IL(0,1)"
            2,"IL(0,0)"
            3,"IL(0,1)"
            4,"IL(0,0)"
            5,"IL(0,1)"
            6,"IL(0,1)""#
        );

        let codebook_data = indoc!(
            "
            ---
            table_name: test
            col_metadata:
              - name: x
                coltype:
                  Labeler:
                    n_labels: 2 
                    pr_h:
                      a: 1.0
                      b: 2.0
                    pr_k:
                      a: 3.0
                      b: 4.0
                    pr_world:
                      alpha: 1.0
                      k: 2
            "
        );

        let codebook: Codebook = serde_yaml::from_str(&codebook_data).unwrap();

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let col_models = read_cols(reader, &codebook).unwrap();

        let prior = match &col_models[0] {
            ColModel::Labeler(ftr) => ftr.prior.clone(),
            _ => panic!("wrong feature type"),
        };

        assert_relative_eq!(prior.pr_h.a(), 1.0, epsilon = 1E-12);
        assert_relative_eq!(prior.pr_h.b(), 2.0, epsilon = 1E-12);

        assert_relative_eq!(prior.pr_k.a(), 3.0, epsilon = 1E-12);
        assert_relative_eq!(prior.pr_k.b(), 4.0, epsilon = 1E-12);

        assert_relative_eq!(prior.pr_world.alpha(), 2.0, epsilon = 1E-12);
        assert_eq!(prior.pr_world.k(), 2);
    }
}

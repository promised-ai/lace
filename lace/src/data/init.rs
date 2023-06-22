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
use crate::cc::feature::{
    ColModel, Column, Feature, Latent, MissingNotAtRandom,
};
use crate::codebook::{Codebook, ColType};
use crate::codebook::{CodebookError, ValueMap};
use crate::error::DataParseError;
use lace_data::{Container, SparseContainer};
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::rv::dist::{Gamma, NormalInvChiSquared, SymmetricDirichlet};

use polars::prelude::{DataFrame, Series};
use std::collections::HashMap;

fn continuous_col_model<R: rand::Rng>(
    id: usize,
    srs: &Series,
    hyper_opt: Option<NixHyper>,
    prior_opt: Option<NormalInvChiSquared>,
    mut rng: &mut R,
) -> Result<ColModel, CodebookError> {
    let xs: Vec<Option<f64>> =
        crate::codebook::data::series_to_opt_vec!(srs, f64);
    let data = SparseContainer::from(xs);
    let (hyper, prior, ignore_hyper) = match (hyper_opt, prior_opt) {
        (Some(hy), Some(pr)) => (hy, pr, true),
        (Some(hy), None) => {
            let pr = hy.draw(rng);
            (hy, pr, false)
        }
        (None, Some(pr)) => (NixHyper::default(), pr, true),
        (None, None) => {
            let xs = data.present_cloned();
            let hy = NixHyper::from_data(&xs);
            let pr = hy.draw(&mut rng);
            (hy, pr, false)
        }
    };
    let mut col = Column::new(id, data, prior, hyper);
    col.ignore_hyper = ignore_hyper;
    Ok(ColModel::Continuous(col))
}

fn count_col_model<R: rand::Rng>(
    id: usize,
    srs: &Series,
    hyper_opt: Option<PgHyper>,
    prior_opt: Option<Gamma>,
    mut rng: &mut R,
) -> Result<ColModel, CodebookError> {
    let xs: Vec<Option<u32>> =
        crate::codebook::data::series_to_opt_vec!(srs, u32);
    let data = SparseContainer::from(xs);
    let (hyper, prior, ignore_hyper) = match (hyper_opt, prior_opt) {
        (Some(hy), Some(pr)) => (hy, pr, true),
        (Some(hy), None) => {
            let pr = hy.draw(rng);
            (hy, pr, false)
        }
        (None, Some(pr)) => (PgHyper::default(), pr, true),
        (None, None) => {
            let xs = data.present_cloned();
            let hy = PgHyper::from_data(&xs);
            let pr = hy.draw(&mut rng);
            (hy, pr, false)
        }
    };
    let mut col = Column::new(id, data, prior, hyper);
    col.ignore_hyper = ignore_hyper;
    Ok(ColModel::Count(col))
}

fn is_categorical_int_dtype(dtype: &polars::datatypes::DataType) -> bool {
    use polars::datatypes::DataType;
    matches!(
        dtype,
        DataType::Boolean
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
    )
}

fn categorical_col_model<R: rand::Rng>(
    id: usize,
    srs: &Series,
    hyper_opt: Option<CsdHyper>,
    prior_opt: Option<SymmetricDirichlet>,
    k: usize,
    value_map: &ValueMap,
    mut rng: &mut R,
) -> Result<ColModel, CodebookError> {
    use polars::datatypes::DataType;
    let xs: Vec<Option<u8>> = match (value_map, srs.dtype()) {
        (ValueMap::String(map), DataType::Utf8) => {
            crate::codebook::data::series_to_opt_strings!(srs)
                .iter()
                .map(|val| val.as_ref().map(|s| map.ix(s).unwrap() as u8))
                .collect()
        }
        (ValueMap::U8(_), dt) if is_categorical_int_dtype(dt) => {
            crate::codebook::data::series_to_opt_vec!(srs, u8)
        }
        _ => {
            return Err(CodebookError::UnsupportedDataType {
                col_name: srs.name().to_owned(),
                dtype: srs.dtype().clone(),
            })
        }
    };
    let data = SparseContainer::from(xs);
    let (hyper, prior, ignore_hyper) = match (hyper_opt, prior_opt) {
        (Some(hy), Some(pr)) => (hy, pr, true),
        (Some(hy), None) => {
            let pr = hy.draw(k, rng);
            (hy, pr, false)
        }
        (None, Some(pr)) => (CsdHyper::new(1.0, 1.0), pr, true),
        (None, None) => {
            let hy = CsdHyper::new(1.0, 1.0);
            let pr = hy.draw(k, &mut rng);
            (hy, pr, false)
        }
    };
    let mut col = Column::new(id, data, prior, hyper);
    col.ignore_hyper = ignore_hyper;
    Ok(ColModel::Categorical(col))
}

pub fn df_to_col_models<R: rand::Rng>(
    codebook: Codebook,
    df: DataFrame,
    rng: &mut R,
) -> Result<(Codebook, Vec<ColModel>), DataParseError> {
    if !codebook.col_metadata.is_empty() && df.is_empty() {
        return Err(DataParseError::ColumnMetadataSuppliedForEmptyData);
    }
    if !codebook.row_names.is_empty() && df.is_empty() {
        return Err(DataParseError::RowNamesSuppliedForEmptyData);
    }

    if df.is_empty() {
        return Ok((codebook, Vec::new()));
    }

    let id_col = {
        let mut id_cols = df
            .get_column_names()
            .iter()
            .filter(|&name| lace_utils::is_index_col(name))
            .map(|name| name.to_string())
            .collect::<Vec<String>>();

        if id_cols.is_empty() {
            Err(DataParseError::NoIDColumn)
        } else if id_cols.len() > 1 {
            Err(DataParseError::MultipleIdColumns(id_cols))
        } else {
            Ok(id_cols.pop().expect("Should have had one ID column"))
        }
    }?;

    let srss = {
        let mut srss: HashMap<&str, &Series> = df
            .get_columns()
            .iter()
            .map(|srs| (srs.name(), srs))
            .collect();
        srss.remove(id_col.as_str())
            .ok_or(DataParseError::NoIDColumn)?;
        srss
    };

    let col_models: Vec<ColModel> = codebook
        .col_metadata
        .iter()
        .enumerate()
        .map(|(id, colmd)| {
            let srs = srss[colmd.name.as_str()];
            let col_model = match &colmd.coltype {
                ColType::Continuous { hyper, prior } => continuous_col_model(
                    id,
                    srs,
                    hyper.clone(),
                    prior.clone(),
                    rng,
                )
                .map_err(DataParseError::Codebook),
                ColType::Count { hyper, prior } => {
                    count_col_model(id, srs, hyper.clone(), prior.clone(), rng)
                        .map_err(DataParseError::Codebook)
                }
                ColType::Categorical {
                    hyper,
                    prior,
                    k,
                    value_map,
                } => categorical_col_model(
                    id,
                    srs,
                    hyper.clone(),
                    prior.clone(),
                    *k,
                    value_map,
                    rng,
                )
                .map_err(DataParseError::Codebook),
            };

            // If missing not at random, convert the column type
            if colmd.missing_not_at_random {
                use lace_stats::rv::dist::Beta;
                use polars::prelude::DataType;
                col_model.map(|cm| {
                    ColModel::MissingNotAtRandom(MissingNotAtRandom {
                        present: {
                            let prior = Beta::jeffreys();
                            let data = SparseContainer::from(
                                srs.iter()
                                    .map(|x| {
                                        let dtype = x.dtype();
                                        // Unknown type is considered missing
                                        !(matches!(dtype, DataType::Null)
                                            || matches!(
                                                dtype,
                                                DataType::Unknown
                                            ))
                                    })
                                    .collect::<Vec<bool>>(),
                            );
                            Column::new(id, data, prior, ())
                        },
                        fx: Box::new(cm),
                    })
                })
            } else if colmd.latent {
                col_model.map(|cm| {
                    ColModel::Latent(Latent {
                        column: Box::new(cm),
                        assignment: Vec::new(),
                    })
                })
            } else {
                col_model
            }
        })
        .collect::<Result<_, DataParseError>>()?;

    if col_models
        .iter()
        .any(|cm| cm.len() != codebook.row_names.len())
    {
        dbg!(
            col_models.iter().map(|cm| cm.len()).collect::<Vec<_>>(),
            codebook.row_names.len()
        );
        // FIXME!
        // return Err(CsvParseError::CodebookAndDataRowMismatch);
    }
    Ok((codebook, col_models))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use indoc::indoc;

    fn str_to_tempfile(data: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(data.as_bytes()).unwrap();
        f
    }

    #[test]
    fn uses_codebook_continuous_hyper_if_specified() {
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
                  !Continuous
                    hyper:
                      pr_m:
                        mu: 0.0
                        sigma: 1.0
                      pr_k:
                        shape: 2.0
                        rate: 3.0
                      pr_v:
                        shape: 4.0
                        scale: 5.0
                      pr_s2:
                        shape: 6.0
                        scale: 7.0
                missing_not_at_random: false
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

        let codebook: Codebook = serde_yaml::from_str(codebook_data).unwrap();

        let mut rng = rand::thread_rng();

        let file = str_to_tempfile(csv_data);
        let (_, col_models) = df_to_col_models(
            codebook,
            lace_codebook::data::read_csv(file.path()).unwrap(),
            &mut rng,
        )
        .unwrap();

        let hyper = match &col_models[0] {
            ColModel::Continuous(ftr) => ftr.hyper.clone(),
            _ => panic!("wrong feature type"),
        };
        assert_relative_eq!(hyper.pr_m.mu(), 0.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_m.sigma(), 1.0, epsilon = 1E-12);

        assert_relative_eq!(hyper.pr_k.shape(), 2.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_k.rate(), 3.0, epsilon = 1E-12);

        assert_relative_eq!(hyper.pr_v.shape(), 4.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_v.scale(), 5.0, epsilon = 1E-12);

        assert_relative_eq!(hyper.pr_s2.shape(), 6.0, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_s2.scale(), 7.0, epsilon = 1E-12);
    }

    #[test]
    fn uses_codebook_categorical_hyper_if_specified() {
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
                  !Categorical
                    k: 2 
                    hyper:
                      pr_alpha:
                        shape: 1.2
                        scale: 3.4
                    value_map: !u8 2
                missing_not_at_random: false
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

        let codebook: Codebook = serde_yaml::from_str(codebook_data).unwrap();

        let mut rng = rand::thread_rng();

        let file = str_to_tempfile(csv_data);
        let (_, col_models) = df_to_col_models(
            codebook,
            lace_codebook::data::read_csv(file.path()).unwrap(),
            &mut rng,
        )
        .unwrap();

        let hyper = match &col_models[0] {
            ColModel::Categorical(ftr) => ftr.hyper.clone(),
            _ => panic!("wrong feature type"),
        };
        assert_relative_eq!(hyper.pr_alpha.shape(), 1.2, epsilon = 1E-12);
        assert_relative_eq!(hyper.pr_alpha.scale(), 3.4, epsilon = 1E-12);
    }
}

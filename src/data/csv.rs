extern crate csv;
extern crate rand;
extern crate rv;

use std::collections::BTreeMap;
use std::f64;
use std::io::Read;

use self::csv::{Reader, StringRecord};
use self::rv::dist::Gamma;

use cc::codebook::{ColMetadata, ColType, SpecType};
use cc::{Codebook, ColModel, Column, DataContainer};
use data::gmd::process_gmd_csv;
use defaults;
use dist::prior::ng::NigHyper;
use dist::prior::{Csd, Ng};
use misc::{n_unique, parse_result, transpose};

/// Reads the columns of a csv into a vector of `ColModel`.
///
/// Data requirements:
/// - The first row of the csv must have a header
/// - The first column of the csv must be `ID`
/// - All columns in the csv, other than `ID`, must be in the codebook
/// - Missing data are empty cells
pub fn read_cols<R: Read>(
    mut reader: Reader<R>,
    codebook: &Codebook,
) -> Vec<ColModel> {
    let mut rng = rand::thread_rng();
    // We need to sort the column metadatas into the same order as the
    // columns appear in the csv file.
    let colmds = {
        // headers() borrows mutably from the reader, so it has to go in its
        // own scope.
        let csv_header = reader.headers().unwrap();
        colmds_by_heaader(&codebook, &csv_header)
    };

    let mut col_models = init_col_models(&colmds);
    for record in reader.records() {
        col_models = push_row(col_models, record.unwrap());
    }
    // FIXME: Should zip with the codebook and use the proper priors
    col_models.iter_mut().for_each(|col_model| match col_model {
        ColModel::Continuous(ftr) => {
            ftr.prior = Ng::from_data(&ftr.data.data, &mut rng);
        }
        _ => (),
    });
    col_models
}

fn push_row(
    mut col_models: Vec<ColModel>,
    record: StringRecord,
) -> Vec<ColModel> {
    let dummy_f64: f64 = 0.0;
    let dummy_u8: u8 = 0;

    col_models
        .iter_mut()
        .zip(record.iter().skip(1)) // assume id is the first column
        .for_each(|(cm, rec)| {
            match cm {
                ColModel::Continuous(ftr) => {
                    let val_opt = parse_result::<f64>(rec);
                    // TODO: Check for NaN, -Inf, and Inf
                    ftr.data.push(val_opt, dummy_f64);
                }
                ColModel::Categorical(ftr) => {
                    let val_opt = parse_result::<u8>(rec);
                    ftr.data.push(val_opt, dummy_u8);
                }
            }
        });

    col_models
}

fn init_col_models(colmds: &Vec<(usize, ColMetadata)>) -> Vec<ColModel> {
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
                ColType::Binary { .. } => {
                    unimplemented!();
                }
            }
        }).collect()
}

fn colmds_by_heaader(
    codebook: &Codebook,
    csv_header: &StringRecord,
) -> Vec<(usize, ColMetadata)> {
    let mut colmds = codebook.col_metadata.clone();
    let mut output = Vec::new();
    for (ix, col_name) in csv_header.iter().enumerate() {
        if ix == 0 && col_name.to_lowercase() != "id" {
            panic!("First column of csv must be \"ID\" or \"id\"");
        }
        let colmd_opt = colmds.remove(&String::from(col_name));
        match colmd_opt {
            Some(colmd) => output.push((colmd.id, colmd)),
            None => (),
        }
    }
    if !colmds.is_empty() {
        panic!("Failed to retrieve all columns");
    }

    if output.len() < (csv_header.len() - 1) {
        panic!("Columns in the csv (other than ID/id) missing from codebook");
    }
    output
}

// Default codebook generation
// ---------------------------
fn is_categorical(col: &Vec<f64>, cutoff: u8) -> bool {
    // drop nan
    let xs: Vec<f64> =
        col.iter().filter(|x| x.is_finite()).map(|x| *x).collect();
    let all_ints = xs.iter().all(|&x| x.round() == x);
    if !all_ints {
        false
    } else {
        n_unique(&xs, cutoff as usize) <= (cutoff as usize)
    }
}

/// Generates a default codebook from a csv file.
pub fn codebook_from_csv<R: Read>(
    mut reader: Reader<R>,
    cat_cutoff: Option<u8>,
    alpha_prior_opt: Option<Gamma>,
    gmd_reader: Option<Reader<R>>,
) -> Codebook {
    let csv_header = reader.headers().unwrap().clone();
    let gmd = match gmd_reader {
        Some(r) => process_gmd_csv(r),
        None => BTreeMap::new(),
    };

    // Load everything into a vec of f64
    let mut row_names: Vec<String> = vec![];
    let data_cols = {
        let f64_data: Vec<Vec<f64>> = reader
            .records()
            .map(|rec| {
                let rec_uw = rec.unwrap();
                let row_name: String = String::from(rec_uw.get(0).unwrap());
                row_names.push(row_name);
                rec_uw
                    .iter()
                    .skip(1)
                    .map(|entry| match parse_result::<f64>(&entry) {
                        Some(x) => x,
                        None => f64::NAN,
                    }).collect()
            }).collect();

        transpose(&f64_data)
    };

    let cutoff = cat_cutoff.unwrap_or(20);
    let mut colmd: BTreeMap<String, ColMetadata> = BTreeMap::new();
    data_cols
        .iter()
        .zip(csv_header.iter().skip(1))
        .enumerate()
        .for_each(|(id, (col, name))| {
            let col_is_categorical = is_categorical(col, cutoff);

            let spec_type = if col_is_categorical {
                match gmd.get(name) {
                    Some(gmd_row) => SpecType::Genotype {
                        chrom: gmd_row.chrom,
                        pos: gmd_row.pos,
                    },
                    None => SpecType::Other,
                }
            } else {
                SpecType::Phenotype
            };

            let coltype = if col_is_categorical {
                let max: f64 = col
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold(0.0, |max, x| if max < *x { *x } else { max });
                let k = (max + 1.0) as usize;
                ColType::Categorical {
                    k,
                    hyper: None,
                    value_map: None,
                }
            } else {
                ColType::Continuous { hyper: None }
            };

            let name = String::from(name);
            let md = ColMetadata {
                id,
                spec_type,
                name: name.clone(),
                coltype,
            };

            colmd.insert(name, md);
        });

    let alpha_prior = alpha_prior_opt.unwrap_or(defaults::GENERAL_ALPHA_PRIOR);

    Codebook {
        table_name: String::from("my_data"),
        view_alpha_prior: Some(alpha_prior.clone()),
        state_alpha_prior: Some(alpha_prior),
        col_metadata: colmd,
        comments: Some(String::from("Auto-generated codebook")),
    }
}

#[cfg(test)]
mod tests {
    use self::csv::ReaderBuilder;
    use super::*;
    use cc::codebook::ColMetadata;
    use cc::SpecType;
    use std::path::Path;

    fn get_codebook() -> Codebook {
        Codebook {
            view_alpha_prior: None,
            state_alpha_prior: None,
            comments: None,
            table_name: String::from("test"),
            col_metadata: btreemap!(
                String::from("y") => ColMetadata {
                    id: 1,
                    spec_type: SpecType::Other,
                    name: String::from("y"),
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                },
                String::from("x") => ColMetadata {
                    id: 0,
                    spec_type: SpecType::Other,
                    name: String::from("x"),
                    coltype: ColType::Continuous { hyper: None },
                },
            ),
        }
    }

    fn data_with_no_missing() -> (String, Codebook) {
        let data = "ID,x,y\n0,0.1,0\n1,1.2,0\n2,2.3,1";
        // NOTE that the metadatas and the csv column names are in different
        // order
        (String::from(data), get_codebook())
    }

    fn data_with_some_missing() -> (String, Codebook) {
        let data = "ID,x,y\n0,,0\n1,1.2,0\n2,2.3,";
        // NOTE that the metadatas and the csv column names are in different
        // order
        (String::from(data), get_codebook())
    }

    #[test]
    fn col_mds_by_header_should_match_header_order() {
        let (data, codebook) = data_with_no_missing();
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let csv_header = &reader.headers().unwrap();
        let colmds = colmds_by_heaader(&codebook, &csv_header);

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
        let colmds = colmds_by_heaader(&codebook, &csv_header);
        let col_models = init_col_models(&colmds);

        assert!(col_models[0].is_continuous());
        assert!(col_models[1].is_categorical());
    }

    #[test]
    fn read_cols_standard_data() {
        let (data, codebook) = data_with_no_missing();
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let col_models = read_cols(reader, &codebook);

        assert!(col_models[0].is_continuous());
        assert!(col_models[1].is_categorical());

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
    fn read_cols_missing_data() {
        let (data, codebook) = data_with_some_missing();
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(data.as_bytes());

        let col_models = read_cols(reader, &codebook);

        assert!(col_models[0].is_continuous());
        assert!(col_models[1].is_categorical());

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
    fn non_rounded_vec_should_be_continuous_regardles_of_cutoff() {
        let xs = vec![0.1, 1.2, 2.3, 3.4];
        assert!(!is_categorical(&xs, 20));
        assert!(!is_categorical(&xs, 2));
    }

    #[test]
    fn some_non_rounded_vec_should_be_continuous_regardles_of_cutoff() {
        let xs = vec![0.0, 1.0, 2.3, 3.0];
        assert!(!is_categorical(&xs, 20));
        assert!(!is_categorical(&xs, 2));
    }

    #[test]
    fn all_rounded_vec_should_be_categorical_if_k_less_than_cutoff() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 2.0];

        assert!(is_categorical(&xs, 20));
        assert!(!is_categorical(&xs, 2));
    }

    #[test]
    fn correct_codebook_with_genomic_metadata() {
        let gmd_reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(Path::new("resources/test/genomics-md.csv"))
            .unwrap();

        let csv_reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(Path::new("resources/test/genomics.csv"))
            .unwrap();

        let cb = codebook_from_csv(csv_reader, None, None, Some(gmd_reader));

        let spec_type =
            |col: &str| cb.col_metadata[&String::from(col)].spec_type.clone();

        assert_eq!(
            spec_type("m_0"),
            SpecType::Genotype {
                pos: 0.12,
                chrom: 1
            }
        );
        assert_eq!(
            spec_type("m_1"),
            SpecType::Genotype {
                pos: 0.23,
                chrom: 1
            }
        );
        assert_eq!(
            spec_type("m_2"),
            SpecType::Genotype {
                pos: 0.45,
                chrom: 2
            }
        );
        assert_eq!(
            spec_type("m_3"),
            SpecType::Genotype {
                pos: 0.67,
                chrom: 2
            }
        );
        assert_eq!(
            spec_type("m_4"),
            SpecType::Genotype {
                pos: 0.89,
                chrom: 3
            }
        );
        assert_eq!(
            spec_type("m_5"),
            SpecType::Genotype {
                pos: 1.01,
                chrom: 3
            }
        );
        assert_eq!(
            spec_type("m_6"),
            SpecType::Genotype {
                pos: 1.12,
                chrom: 3
            }
        );
        assert_eq!(
            spec_type("m_7"),
            SpecType::Genotype {
                pos: 1.23,
                chrom: 4
            }
        );
        assert_eq!(spec_type("other"), SpecType::Other);
        assert_eq!(spec_type("t_1"), SpecType::Phenotype);
        assert_eq!(spec_type("t_2"), SpecType::Phenotype);
    }
}

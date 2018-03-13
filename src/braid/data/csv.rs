extern crate csv;
extern crate rand;

use std::io::Read;
use std::str::FromStr;

use self::csv::{Reader, StringRecord};

use cc::{Codebook, ColModel, Column, DataContainer};
use cc::codebook::ColMetadata;
use dist::prior::{CatSymDirichlet, NormalInverseGamma};
use dist::prior::nig::NigHyper;

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
            ftr.prior = NormalInverseGamma::from_data(&ftr.data.data, &mut rng);
        }
        _ => (),
    });
    col_models
}

fn parse_result<T: FromStr>(res: &str) -> Option<T> {
    // For csv, empty cells are considered missing regardless of type
    if res.is_empty() {
        None
    } else {
        match res.parse::<T>() {
            Ok(x) => Some(x),
            Err(_) => panic!("Could not parse \"{}\"", res),
        }
    }
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
                ColModel::Continuous(ftr)  => {
                    let val_opt = parse_result::<f64>(rec);
                    // TODO: Check for NaN, -Inf, and Inf
                    ftr.data.push(val_opt, dummy_f64);
                },
                ColModel::Categorical(ftr) => {
                    let val_opt = parse_result::<u8>(rec);
                    ftr.data.push(val_opt, dummy_u8);
                },
            }
        });

    col_models
}

fn init_col_models(colmds: &Vec<(usize, ColMetadata)>) -> Vec<ColModel> {
    let mut rng = rand::thread_rng();
    colmds
        .iter()
        .map(|(id, colmd)| {
            match colmd {
                // Ignore hypers until all the data are loaded, then we'll
                // re-initialize
                &ColMetadata::Continuous { .. } => {
                    let data = DataContainer::new(vec![]);
                    let prior = {
                        let h = NigHyper::default();
                        NormalInverseGamma::from_hyper(h, &mut rng)
                    };
                    let column = Column::new(*id, data, prior);
                    ColModel::Continuous(column)
                }
                &ColMetadata::Categorical { k, .. } => {
                    let data = DataContainer::new(vec![]);
                    let prior = { CatSymDirichlet::vague(k, &mut rng) };
                    let column = Column::new(*id, data, prior);
                    ColModel::Categorical(column)
                }
                &ColMetadata::Binary { .. } => {
                    unimplemented!();
                }
            }
        })
        .collect()
}

fn colmds_by_heaader(
    codebook: &Codebook,
    csv_header: &StringRecord,
) -> Vec<(usize, ColMetadata)> {
    let mut colmds = codebook.col_metadata_map();
    let mut output = Vec::new();
    for (ix, col_name) in csv_header.iter().enumerate() {
        if ix == 0 && col_name.to_lowercase() != "id" {
            panic!("First column of csv must be \"ID\" or \"id\"");
        }
        let colmd_opt = colmds.remove(col_name);
        match colmd_opt {
            Some(colmd) => output.push(colmd),
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

#[cfg(test)]
mod tests {
    use super::*;
    use cc::codebook::{ColMetadata, MetaData};
    use self::csv::ReaderBuilder;

    fn get_codebook() -> Codebook {
        Codebook {
            row_names: None,
            table_name: String::from("test"),
            metadata: vec![
                MetaData::Column {
                    id: 1,
                    name: String::from("y"),
                    colmd: ColMetadata::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                },
                MetaData::Column {
                    id: 0,
                    name: String::from("x"),
                    colmd: ColMetadata::Continuous { hyper: None },
                },
            ],
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

        assert!(colmds[0].1.is_continuous());
        assert!(colmds[1].1.is_categorical());
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
}

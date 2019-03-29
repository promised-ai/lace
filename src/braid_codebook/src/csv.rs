extern crate braid_stats;
extern crate csv;
extern crate rand;
extern crate rv;

use std::collections::BTreeMap;
use std::f64;
use std::io::Read;

use braid_stats::defaults;
use csv::Reader;
use rv::dist::Gamma;

use crate::codebook::{Codebook, ColMetadata, ColType, SpecType};
use crate::gmd::process_gmd_csv;
use crate::misc::{is_categorical, parse_result, transpose};

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
                    })
                    .collect()
            })
            .collect();

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
                notes: None,
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
        row_names: Some(row_names),
    }
}


#[cfg(test)]
mod tests {
    extern crate maplit;

    use super::*;

    use std::path::Path;

    use csv::ReaderBuilder;
    use crate::codebook::SpecType;

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

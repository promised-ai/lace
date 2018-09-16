extern crate rv;
extern crate serde_yaml;

use misc::minmax;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use self::rv::dist::Gamma;
use dist::prior::csd::CsdHyper;
use dist::prior::ng::NigHyper;

// TODO: restructure (https://gitlab.com/baxe/braid/issues/8)
/// Codebook object for storing information about the dataset
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Codebook {
    /// The name of the table
    pub table_name: String,
    pub metadata: Vec<MetaData>,
    /// String row names. Mainly for use with various external APIs.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub row_names: Option<Vec<String>>,
    /// Optional misc comments
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub comments: Option<String>,
}

impl Default for Codebook {
    fn default() -> Codebook {
        Codebook::new(String::from("braid"), vec![])
    }
}

impl Codebook {
    pub fn new(table_name: String, metadata: Vec<MetaData>) -> Self {
        Codebook {
            table_name: table_name,
            metadata: metadata,
            row_names: None,
            comments: None,
        }
    }

    pub fn from_yaml(path: &str) -> Self {
        let mut file = File::open(Path::new(&path))
            .expect(format!("Failed to open file {}", path).as_str());
        let mut yaml = String::new();
        file.read_to_string(&mut yaml)
            .expect(format!("Failed to read {} to string", path).as_str());
        serde_yaml::from_str(&yaml).unwrap()
    }

    // FIXME: change to validate IDs
    pub fn validate_ids(&self) -> Result<(), &str> {
        let mut ids: Vec<usize> = Vec::new();
        for md in &self.metadata {
            match md {
                &MetaData::Column { ref id, .. } if ids.contains(&id) => {
                    return Err("IDs not unique")
                }
                &MetaData::Column { ref id, .. } => ids.push(*id),
                _ => (),
            }
        }

        if ids.is_empty() {
            Err("No column metadata")
        } else {
            let (min_id, max_id) = minmax(&ids);
            if min_id != 0 || max_id != ids.len() - 1 {
                Err("IDs must span 0, 1, ..., n_cols-1")
            } else {
                Ok(())
            }
        }
    }

    pub fn zip_col_metadata(&self) -> Vec<(usize, String, ColMetadata)> {
        let mut output = Vec::new();
        for md in &self.metadata {
            match md {
                &MetaData::Column {
                    ref id,
                    ref name,
                    ref colmd,
                    ..
                } => {
                    output.push((*id, name.clone(), colmd.clone()));
                }
                _ => (),
            }
        }
        output.sort_by_key(|(id, _, _)| *id);
        output
    }

    pub fn col_metadata_map(&self) -> BTreeMap<String, (usize, ColMetadata)> {
        let mut output = BTreeMap::new();
        for md in &self.metadata {
            match md {
                &MetaData::Column {
                    ref id,
                    ref name,
                    ref colmd,
                    ..
                } => {
                    output.insert(name.clone(), (*id, colmd.clone()));
                }
                _ => (),
            }
        }
        output
    }

    pub fn get_col_metadata(&self, col: String) -> Option<MetaData> {
        for md in self.metadata.iter() {
            match md {
                MetaData::Column { ref name, .. } => {
                    if *name == col {
                        return Some(md.clone());
                    }
                }
                _ => (),
            }
        }
        None
    }

    pub fn get_state_alpha_prior(&self) -> Option<Gamma> {
        let alpha_opt = self.metadata.iter().find(|md| match md {
            MetaData::StateAlpha { .. } => true,
            _ => false,
        });
        match alpha_opt {
            Some(MetaData::StateAlpha { shape, rate }) => {
                Some(Gamma::new(*shape, *rate).unwrap())
            }
            Some(_) => panic!("Found wrong type"),
            None => None,
        }
    }

    pub fn get_view_alpha_prior(&self) -> Option<Gamma> {
        let alpha_opt = self.metadata.iter().find(|md| match md {
            MetaData::ViewAlpha { .. } => true,
            _ => false,
        });
        match alpha_opt {
            Some(MetaData::ViewAlpha { shape, rate }) => {
                Some(Gamma::new(*shape, *rate).unwrap())
            }
            Some(_) => panic!("Found wrong type"),
            None => None,
        }
    }

    pub fn ncols(&self) -> usize {
        self.metadata.iter().fold(
            0,
            |acc, md| {
                if md.is_column() {
                    acc + 1
                } else {
                    acc
                }
            },
        )
    }

    /// Add the columns of the other codebook into this codebook. Returns a
    /// map, indexed by the old column IDs, containing the new IDs.
    pub fn merge_cols(&mut self, other: &Codebook) -> BTreeMap<usize, usize> {
        let start_id = self.ncols();
        let mds: Vec<&MetaData> = other
            .metadata
            .iter()
            .filter_map(|md| if md.is_column() { Some(md) } else { None })
            .collect();

        let mut id_map: BTreeMap<usize, usize> = BTreeMap::new();
        mds.iter().enumerate().for_each(|(i, md)| {
            match md {
                MetaData::Column {
                    id,
                    name,
                    spec_type,
                    colmd,
                } => {
                    // TODO: Return result instead of panicing
                    if self.get_col_metadata(name.clone()).is_some() {
                        panic!("Duplicate column name in second codebook");
                    }
                    let new_id = start_id + i;
                    let newmd = MetaData::Column {
                        id: new_id,
                        name: name.clone(),
                        spec_type: spec_type.clone(),
                        colmd: colmd.clone(),
                    };
                    self.metadata.push(newmd);
                    id_map.insert(*id, new_id);
                }
                _ => (),
            }
        });

        id_map
    }

    pub fn reindex_cols(&mut self, id_map: &BTreeMap<usize, usize>) {
        self.metadata.iter_mut().for_each(|md| match md {
            MetaData::Column { id, .. } => *id = id_map[id],
            _ => (),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ColMetadata {
    Continuous {
        #[serde(default)]
        hyper: Option<NigHyper>,
    },
    Categorical {
        k: usize,
        #[serde(default)]
        hyper: Option<CsdHyper>,
        #[serde(default)]
        value_map: Option<BTreeMap<usize, String>>,
    },
    Binary {
        a: f64,
        b: f64,
    },
}

impl ColMetadata {
    pub fn is_continuous(&self) -> bool {
        match self {
            ColMetadata::Continuous { .. } => true,
            _ => false,
        }
    }

    pub fn is_categorical(&self) -> bool {
        match self {
            ColMetadata::Categorical { .. } => true,
            _ => false,
        }
    }

    pub fn is_binary(&self) -> bool {
        match self {
            ColMetadata::Binary { .. } => true,
            _ => false,
        }
    }
}

/// Special type of data. Specifies model-specific type information. Intended
/// to be used with model-specific braid clients.
#[derive(PartialEq, Serialize, Deserialize, Debug, Clone, Copy)]
pub enum SpecType {
    /// Genetic marker type with a chromosome number and position in cM
    Genotype {
        chrom: u8,
        pos: f64,
    },
    /// Phenotype or trait
    Phenotype,
    /// A variable that likely affects the phenotype
    Environmental,
    Other,
}

impl Default for SpecType {
    fn default() -> Self {
        SpecType::Other
    }
}

impl SpecType {
    pub fn is_other(&self) -> bool {
        match self {
            SpecType::Other => true,
            _ => false,
        }
    }
}

// TODO: Restructure (https://gitlab.com/baxe/braid/issues/8)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MetaData {
    Column {
        id: usize,
        name: String,
        // #[serde(skip_serializing_if = "SpecType::is_other")]
        #[serde(default)]
        spec_type: SpecType,
        colmd: ColMetadata,
    },
    StateAlpha {
        shape: f64,
        rate: f64,
    },
    ViewAlpha {
        shape: f64,
        rate: f64,
    },
}

impl MetaData {
    pub fn is_column(&self) -> bool {
        match &self {
            MetaData::Column { .. } => true,
            _ => false,
        }
    }

    pub fn is_state_alpha(&self) -> bool {
        match &self {
            MetaData::StateAlpha { .. } => true,
            _ => false,
        }
    }

    pub fn is_view_alpha(&self) -> bool {
        match &self {
            MetaData::ViewAlpha { .. } => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_codebook() -> Codebook {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };
        let md1 = MetaData::Column {
            id: 1,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            colmd: colmd.clone(),
        };
        let md2 = MetaData::Column {
            id: 2,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            colmd: colmd.clone(),
        };
        let md3 = MetaData::StateAlpha {
            shape: 2.0,
            rate: 3.0,
        };
        let md4 = MetaData::ViewAlpha {
            shape: 2.0,
            rate: 3.0,
        };

        let metadata = vec![md0, md1, md2, md3, md4];
        Codebook::new("table".to_string(), metadata)
    }

    #[test]
    fn validate_ids_with_properly_formed_ids_should_pass() {
        let codebook = quick_codebook();
        assert!(codebook.validate_ids().is_ok());
    }

    #[test]
    fn validate_ids_with_duplicates_should_fail() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };
        let md1 = MetaData::Column {
            id: 2,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            colmd: colmd.clone(),
        };
        let md2 = MetaData::Column {
            id: 2,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            colmd: colmd.clone(),
        };
        let md3 = MetaData::StateAlpha {
            shape: 2.0,
            rate: 3.0,
        };
        let md4 = MetaData::ViewAlpha {
            shape: 2.0,
            rate: 3.0,
        };

        let metadata = vec![md0, md1, md2, md3, md4];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_one_column_should_pass_if_id_is_0() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };

        let metadata = vec![md0];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_ok());
    }

    #[test]
    fn validate_ids_with_one_column_should_fail_if_id_is_not_0() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 1,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };

        let metadata = vec![md0];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_no_columns_should_pfail() {
        let md0 = MetaData::StateAlpha {
            shape: 2.0,
            rate: 3.0,
        };
        let md1 = MetaData::ViewAlpha {
            shape: 2.0,
            rate: 3.0,
        };

        let metadata = vec![md0, md1];

        let codebook = Codebook::new("table".to_string(), metadata);
        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_bad_id_span_should_fail_1() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 1,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };
        let md1 = MetaData::Column {
            id: 2,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            colmd: colmd.clone(),
        };
        let md2 = MetaData::Column {
            id: 3,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            colmd: colmd.clone(),
        };

        let metadata = vec![md0, md1, md2];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_bad_id_span_should_fail_2() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };
        let md1 = MetaData::Column {
            id: 1,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            colmd: colmd.clone(),
        };
        let md2 = MetaData::Column {
            id: 3,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            colmd: colmd.clone(),
        };

        let metadata = vec![md0, md1, md2];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn zip_col_metadata_should_return_an_etry_for_each_column() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };
        let md1 = MetaData::Column {
            id: 1,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            colmd: colmd.clone(),
        };
        let md2 = MetaData::StateAlpha {
            shape: 2.0,
            rate: 3.0,
        };
        let md3 = MetaData::ViewAlpha {
            shape: 2.0,
            rate: 3.0,
        };

        let metadata = vec![md1, md2, md0, md3];

        let codebook = Codebook::new("table".to_string(), metadata);
        let colmds = codebook.zip_col_metadata();

        assert_eq!(colmds.len(), 2);
        assert_eq!(colmds[0].0, 0);
        assert_eq!(colmds[1].0, 1);
    }

    #[test]
    fn get_alphas() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            colmd: colmd.clone(),
        };
        let md1 = MetaData::Column {
            id: 2,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            colmd: colmd.clone(),
        };
        let md2 = MetaData::StateAlpha {
            shape: 2.0,
            rate: 1.0,
        };
        let md3 = MetaData::ViewAlpha {
            shape: 2.0,
            rate: 3.0,
        };

        let metadata = vec![md0, md1, md2, md3];

        let codebook = Codebook::new("table".to_string(), metadata);

        let state_prior_opt = codebook.get_state_alpha_prior();
        assert!(state_prior_opt.is_some());
        let state_prior = state_prior_opt.unwrap();
        assert_relative_eq!(state_prior.shape, 2.0, epsilon = 10E-10);
        assert_relative_eq!(state_prior.rate, 1.0, epsilon = 10E-10);

        let view_prior_opt = codebook.get_view_alpha_prior();
        assert!(view_prior_opt.is_some());
        let view_prior = view_prior_opt.unwrap();
        assert_relative_eq!(view_prior.shape, 2.0, epsilon = 10E-10);
        assert_relative_eq!(view_prior.rate, 3.0, epsilon = 10E-10);
    }

    #[test]
    fn ncols_returns_number_of_column_metadata() {
        let cb = quick_codebook();
        assert_eq!(cb.ncols(), 3);
    }

    #[test]
    fn merge_codebooks_produces_correct_ids() {
        let mut cb1 = quick_codebook();
        let cb2 = {
            let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
            let md0 = MetaData::Column {
                id: 0,
                spec_type: SpecType::Other,
                name: "fwee".to_string(),
                colmd: colmd.clone(),
            };
            let md1 = MetaData::Column {
                id: 1,
                spec_type: SpecType::Other,
                name: "four".to_string(),
                colmd: colmd.clone(),
            };
            let metadata = vec![md0, md1];
            Codebook::new("table2".to_string(), metadata)
        };
        cb1.merge_cols(&cb2);

        assert_eq!(cb1.ncols(), 5);

        let colmds = cb1.col_metadata_map();
        assert_eq!(colmds.len(), 5);
        assert_eq!(colmds.get(&"fwee".to_string()).unwrap().0, 3);
        assert_eq!(colmds.get(&"four".to_string()).unwrap().0, 4);
    }
}

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use braid_stats::prior::{CrpPrior, CsdHyper, NigHyper};
use braid_utils::misc::minmax;
use maplit::btreemap;
use rv::dist::Kumaraswamy;
use serde::{Deserialize, Serialize};

/// Codebook object for storing information about the dataset
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub struct Codebook {
    /// The name of the table
    pub table_name: String,
    /// Prior on State CRP alpha parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub state_alpha_prior: Option<CrpPrior>,
    /// Prior on View CRP alpha parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub view_alpha_prior: Option<CrpPrior>,
    /// The metadata for each column indexed by name
    pub col_metadata: BTreeMap<String, ColMetadata>,
    /// Optional misc comments
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub comments: Option<String>,
    /// Optional names of each row
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub row_names: Option<Vec<String>>,
}

impl Default for Codebook {
    fn default() -> Codebook {
        Codebook::new(String::from("braid"), btreemap!())
    }
}

impl Codebook {
    pub fn new(
        table_name: String,
        col_metadata: BTreeMap<String, ColMetadata>,
    ) -> Self {
        Codebook {
            table_name,
            col_metadata,
            view_alpha_prior: None,
            state_alpha_prior: None,
            comments: None,
            row_names: None,
        }
    }

    pub fn from_yaml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut yaml = String::new();
        file.read_to_string(&mut yaml)?;
        let codebook: Codebook = serde_yaml::from_str(&yaml).map_err(|_| {
            let err_kind = io::ErrorKind::InvalidData;
            io::Error::new(err_kind, "Failed to parse file into codebook")
        })?;
        Ok(codebook)
    }

    // TODO: change to validate IDs
    pub fn validate_ids(&self) -> Result<(), &str> {
        let mut ids: Vec<usize> = Vec::new();
        let duplicate_ids = self
            .col_metadata
            .values()
            .find(|colmd| {
                if ids.contains(&colmd.id) {
                    true
                } else {
                    ids.push(colmd.id);
                    false
                }
            })
            .is_some();

        if duplicate_ids {
            return Err("duplicate IDs found");
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

    /// Return a vector of tuples containing the column ID, the column name,
    /// and the column metadata, sorted in ascending order by ID.
    pub fn zip_col_metadata(&self) -> Vec<(usize, String, ColMetadata)> {
        let mut output: Vec<(usize, String, ColMetadata)> = self
            .col_metadata
            .values()
            .map(|colmd| (colmd.id, colmd.name.clone(), colmd.clone()))
            .collect();
        output.sort_by_key(|(id, _, _)| *id);
        output
    }

    pub fn col_metadata(&self, col: String) -> Option<&ColMetadata> {
        self.col_metadata.get(&col)
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.col_metadata.len()
    }

    /// Add the columns of the other codebook into this codebook. Returns a
    /// map, indexed by the old column IDs, containing the new IDs.
    pub fn merge_cols(&mut self, other: &Codebook) -> BTreeMap<usize, usize> {
        let start_id = self.ncols();
        let mut id_map: BTreeMap<usize, usize> = BTreeMap::new();
        other
            .col_metadata
            .values()
            .enumerate()
            .for_each(|(i, colmd)| {
                // TODO: Return result instead of panicing
                if self.col_metadata.contains_key(&colmd.name) {
                    panic!("Duplicate column name in second codebook");
                }
                let new_id = start_id + i;
                let newmd = ColMetadata {
                    id: new_id,
                    name: colmd.name.clone(),
                    spec_type: colmd.spec_type.clone(),
                    coltype: colmd.coltype.clone(),
                    notes: colmd.notes.clone(),
                };
                self.col_metadata.insert(colmd.name.clone(), newmd);
                id_map.insert(colmd.id, new_id);
            });

        id_map
    }

    pub fn reindex_cols(&mut self, id_map: &BTreeMap<usize, usize>) {
        self.col_metadata.values_mut().for_each(|colmd| {
            let id = colmd.id;
            colmd.id = id_map[&id];
        })
    }
}

/// Stores metadata for the specific column types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub enum ColType {
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
    Labeler {
        #[serde(default)]
        pr_h: Option<Kumaraswamy>,
        #[serde(default)]
        pr_k: Option<Kumaraswamy>,
        #[serde(default)]
        pr_world: Option<Kumaraswamy>,
    },
}

impl ColType {
    pub fn is_continuous(&self) -> bool {
        match self {
            ColType::Continuous { .. } => true,
            _ => false,
        }
    }

    pub fn is_categorical(&self) -> bool {
        match self {
            ColType::Categorical { .. } => true,
            _ => false,
        }
    }

    pub fn is_labeler(&self) -> bool {
        match self {
            ColType::Labeler { .. } => true,
            _ => false,
        }
    }
}

/// Special type of data. Specifies model-specific type information. Intended
/// to be used with model-specific braid clients.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, PartialOrd)]
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub struct ColMetadata {
    /// Column index. Columns should have unique IDs in 0, .., n-1
    pub id: usize,
    /// The name of the Column
    pub name: String,
    #[serde(default)]
    pub spec_type: SpecType,
    /// The column model
    pub coltype: ColType,
    #[serde(default)]
    /// Optional notes about the column
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use maplit::{btreemap, convert_args};

    fn quick_codebook() -> Codebook {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            id: 1,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
            id: 2,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = convert_args!(
            keys = String::from,
            btreemap! (
                "0" => md0,
                "1" => md1,
                "2" => md2,
            )
        );
        Codebook::new("table".to_string(), col_metadata)
    }

    #[test]
    fn validate_ids_with_properly_formed_ids_should_pass() {
        let codebook = quick_codebook();
        assert!(codebook.validate_ids().is_ok());
    }

    #[test]
    fn validate_ids_with_duplicates_should_fail() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            id: 2,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
            id: 2,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = convert_args!(
            keys = String::from,
            btreemap! (
                "0" => md0,
                "1" => md1,
                "2" => md2,
            )
        );

        let codebook = Codebook::new("table".to_string(), col_metadata);

        // FIXME: this fails
        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_one_column_should_pass_if_id_is_0() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = btreemap!(String::from("0") => md0);
        let codebook = Codebook::new("table".to_string(), col_metadata);

        assert!(codebook.validate_ids().is_ok());
    }

    #[test]
    fn validate_ids_with_one_column_should_fail_if_id_is_not_0() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            id: 1,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = btreemap!( String::from("0") => md0 );
        let codebook = Codebook::new("table".to_string(), col_metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_bad_id_span_should_fail_1() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            id: 1,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            id: 2,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
            id: 3,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = convert_args!(
            keys = String::from,
            btreemap!(
                "0" => md0,
                "1" => md1,
                "2" => md2,
            )
        );
        let codebook = Codebook::new("table".to_string(), col_metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_bad_id_span_should_fail_2() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            id: 0,
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            id: 1,
            spec_type: SpecType::Other,
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
            id: 3,
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = convert_args!(
            keys = String::from,
            btreemap!(
                "0" => md0,
                "1" => md1,
                "2" => md2,
            )
        );
        let codebook = Codebook::new("table".to_string(), col_metadata);

        assert!(codebook.validate_ids().is_err());
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
            let coltype = ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            };
            let md0 = ColMetadata {
                id: 0,
                spec_type: SpecType::Other,
                name: "fwee".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let md1 = ColMetadata {
                id: 1,
                spec_type: SpecType::Other,
                name: "four".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let col_metadata = btreemap!(
                String::from("fwee") => md0,
                String::from("four") => md1
            );
            Codebook::new("table2".to_string(), col_metadata)
        };
        cb1.merge_cols(&cb2);

        assert_eq!(cb1.ncols(), 5);

        let colmds = cb1.col_metadata;
        assert_eq!(colmds.len(), 5);
        // The btreemap sorts the indice, so 'four' comes before 'fwee'
        assert_eq!(colmds.get(&"fwee".to_string()).unwrap().id, 4);
        assert_eq!(colmds.get(&"four".to_string()).unwrap().id, 3);
    }
}

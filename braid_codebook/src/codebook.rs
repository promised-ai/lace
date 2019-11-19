use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use super::error::MergeColumnsError;
use braid_stats::prior::{CrpPrior, CsdHyper, NigHyper};
use braid_utils::ForEachOk;
use rv::dist::{Kumaraswamy, SymmetricDirichlet};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(into = "Vec<ColMetadata>")]
pub struct ColMetadataList {
    metadata: Vec<ColMetadata>,
    index_lookup: HashMap<String, usize>,
}

impl ColMetadataList {
    pub fn new(metadata: Vec<ColMetadata>) -> Result<Self, String> {
        let mut index_lookup = HashMap::new();
        metadata
            .iter()
            .enumerate()
            .for_each_ok(|(ix, md)| {
                if index_lookup.insert(md.name.clone(), ix).is_none() {
                    Ok(())
                } else {
                    Err(md.name.clone())
                }
            })
            .map(|_| ColMetadataList {
                metadata,
                index_lookup,
            })
    }

    pub fn push(&mut self, md: ColMetadata) -> Result<(), String> {
        use std::collections::hash_map::Entry;

        let n = self.len();
        match self.index_lookup.entry(md.name.clone()) {
            Entry::Vacant(entry) => {
                self.metadata.push(md);
                entry.insert(n);
                debug_assert_eq!(self.metadata.len(), self.index_lookup.len());
                Ok(())
            }
            _ => Err(md.name.clone()),
        }
    }

    pub fn iter(&self) -> std::slice::Iter<ColMetadata> {
        self.metadata.iter()
    }

    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    pub fn contains_key(&self, name: &String) -> bool {
        self.index_lookup.contains_key(name)
    }

    pub fn get(&self, name: &String) -> Option<(usize, &ColMetadata)> {
        self.index_lookup
            .get(name)
            .map(|&ix| (ix, &self.metadata[ix]))
    }
}

impl Into<Vec<ColMetadata>> for ColMetadataList {
    fn into(self) -> Vec<ColMetadata> {
        self.metadata
    }
}

impl std::ops::Index<usize> for ColMetadataList {
    type Output = ColMetadata;

    fn index(&self, ix: usize) -> &Self::Output {
        &self.metadata[ix]
    }
}

impl Default for ColMetadataList {
    fn default() -> Self {
        ColMetadataList {
            metadata: Vec::new(),
            index_lookup: HashMap::new(),
        }
    }
}

impl TryInto<ColMetadataList> for Vec<ColMetadata> {
    type Error = String;

    fn try_into(self) -> Result<ColMetadataList, Self::Error> {
        ColMetadataList::new(self)
    }
}

mod cmlist_serde {
    use super::*;
    use serde::de::{Deserializer, Error};

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<ColMetadataList, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<ColMetadata>::deserialize(deserializer).and_then(|mds| {
            ColMetadataList::new(mds).map_err(|dup_col| {
                Error::custom(format!("Duplicate column name: {}", dup_col))
            })
        })
    }
}

/// Codebook object for storing information about the dataset
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
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
    #[serde(deserialize_with = "cmlist_serde::deserialize")]
    pub col_metadata: ColMetadataList,
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
        Codebook::new(String::from("braid"), ColMetadataList::default())
    }
}

impl Codebook {
    pub fn new(table_name: String, col_metadata: ColMetadataList) -> Self {
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

    /// Return a vector of tuples containing the column ID, the column name,
    /// and the column metadata, sorted in ascending order by ID.
    pub fn zip_col_metadata(&self) -> Vec<(usize, String, ColMetadata)> {
        let mut output: Vec<(usize, String, ColMetadata)> = self
            .col_metadata
            .iter()
            .enumerate()
            .map(|(id, colmd)| (id, colmd.name.clone(), colmd.clone()))
            .collect();
        output.sort_by_key(|(id, _, _)| *id);
        output
    }

    pub fn col_metadata(&self, col: String) -> Option<&ColMetadata> {
        // self.col_metadata.get(&col)
        self.col_metadata.iter().find(|md| md.name == col)
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.col_metadata.len()
    }

    /// Add the columns of the other codebook into this codebook. Returns a
    /// map, indexed by the old column IDs, containing the new IDs.
    pub fn merge_cols(
        &mut self,
        other: Codebook,
    ) -> Result<(), MergeColumnsError> {
        let mut new_col_metadata: Vec<_> = other.col_metadata.into();

        new_col_metadata.drain(..).for_each_ok(|colmd| {
            self.col_metadata.push(colmd).map_err(|name| {
                MergeColumnsError::DuplicateColumnNameError(name)
            })
        })
    }
}

/// Stores metadata for the specific column types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
pub enum ColType {
    /// Univariate continuous (Gaussian) data model
    Continuous {
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        hyper: Option<NigHyper>,
    },
    /// Categorical data up to 256 instances
    Categorical {
        k: usize,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        hyper: Option<CsdHyper>,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        value_map: Option<BTreeMap<usize, String>>,
    },
    /// Human-labeled categorical data
    Labeler {
        n_labels: u8,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pr_h: Option<Kumaraswamy>,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pr_k: Option<Kumaraswamy>,
        #[serde(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pr_world: Option<SymmetricDirichlet>,
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

    /// Return the value map if the type is categorical and a value map exists.
    pub fn value_map(&self) -> Option<&BTreeMap<usize, String>> {
        match self {
            ColType::Categorical { value_map, .. } => value_map.as_ref(),
            _ => None,
        }
    }

    /// Return the index lookup which looks up the value index given the value
    /// String.
    pub fn lookup(&self) -> Option<HashMap<String, usize>> {
        self.value_map().map(|value_map| {
            let mut lookup: HashMap<String, usize> = HashMap::new();
            for (&ix, val) in value_map.iter() {
                lookup.insert(val.clone(), ix);
            }
            lookup
        })
    }
}

/// Special type of data. Specifies model-specific type information. Intended
/// to be used with model-specific braid clients.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
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

/// The metadata associated with a column. In addition to the id and name, it
/// contains information about the data model of a column.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
pub struct ColMetadata {
    /// The name of the Column
    pub name: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "SpecType::is_other")]
    pub spec_type: SpecType,
    /// The column model
    pub coltype: ColType,
    /// Optional notes about the column
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    fn quick_codebook() -> Codebook {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            spec_type: SpecType::Other,
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = ColMetadataList::new(vec![md0, md1, md2]).unwrap();
        Codebook::new("table".to_string(), col_metadata)
    }

    #[test]
    fn new_with_duplicate_names_should_fail() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            spec_type: SpecType::Other,
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
            spec_type: SpecType::Other,
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };

        let col_metadata = ColMetadataList::new(vec![md0, md1, md2]);

        assert_eq!(col_metadata, Err(String::from("2")));
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
                spec_type: SpecType::Other,
                name: "fwee".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let md1 = ColMetadata {
                spec_type: SpecType::Other,
                name: "four".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let col_metadata = ColMetadataList::new(vec![md0, md1]).unwrap();
            Codebook::new("table2".to_string(), col_metadata)
        };

        cb1.merge_cols(cb2).unwrap();
        assert_eq!(cb1.ncols(), 5);

        assert_eq!(cb1.col_metadata[0].name, String::from("0"));
        assert_eq!(cb1.col_metadata[1].name, String::from("1"));
        assert_eq!(cb1.col_metadata[2].name, String::from("2"));

        assert_eq!(cb1.col_metadata[3].name, String::from("fwee"));
        assert_eq!(cb1.col_metadata[4].name, String::from("four"));
    }

    #[test]
    fn merge_cols_detects_duplicate_columns() {
        let mut cb1 = quick_codebook();
        let cb2 = {
            let coltype = ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: None,
            };
            let md0 = ColMetadata {
                spec_type: SpecType::Other,
                name: "1".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let md1 = ColMetadata {
                spec_type: SpecType::Other,
                name: "four".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let col_metadata = ColMetadataList::new(vec![md0, md1]).unwrap();
            Codebook::new("table2".to_string(), col_metadata)
        };

        match cb1.merge_cols(cb2) {
            Err(MergeColumnsError::DuplicateColumnNameError(col)) => {
                assert_eq!(col, String::from("1"))
            }
            Ok(_) => panic!("merge should have detected duplicate"),
        }
    }

    #[test]
    fn lookup_for_continuous_coltype_is_none() {
        let coltype = ColType::Continuous { hyper: None };
        assert!(coltype.lookup().is_none());
    }

    #[test]
    fn lookup_for_labeler_coltype_is_none() {
        let coltype = ColType::Labeler {
            n_labels: 2,
            pr_h: None,
            pr_k: None,
            pr_world: None,
        };
        assert!(coltype.lookup().is_none());
    }

    #[test]
    fn lookup_for_empty_categorical_coltype_is_none() {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            value_map: None,
        };
        assert!(coltype.lookup().is_none());
    }

    #[test]
    fn lookup_for_categorical_coltype_check() {
        let mut value_map: BTreeMap<usize, String> = BTreeMap::new();
        value_map.insert(0, String::from("dog"));
        value_map.insert(1, String::from("cat"));
        value_map.insert(2, String::from("hamster"));
        let coltype = ColType::Categorical {
            k: 3,
            hyper: None,
            value_map: Some(value_map),
        };
        if let Some(lookup) = coltype.lookup() {
            assert_eq!(lookup.len(), 3);
            assert_eq!(lookup.get(&String::from("dog")), Some(&0_usize));
            assert_eq!(lookup.get(&String::from("cat")), Some(&1_usize));
            assert_eq!(lookup.get(&String::from("hamster")), Some(&2_usize));
            assert_eq!(lookup.get(&String::from("gerbil")), None);
        } else {
            assert!(false)
        }
    }

    #[test]
    fn deserialize_metadata_list() {
        let raw = indoc!(
            r#"
            ---
            table_name: my-table
            col_metadata:
              - name: one
                coltype:
                  Continuous:
                    hyper: ~
              - name: two
                coltype:
                  Categorical:
                    k: 2
              - name: three
                coltype:
                  Categorical:
                    k: 2
            "#
        );
        let cb: Codebook = serde_yaml::from_str(&raw).unwrap();
        assert_eq!(cb.col_metadata.len(), 3);
    }

    #[test]
    #[should_panic]
    fn deserialize_metadata_list_with_duplicate_names_fails() {
        let raw = indoc!(
            r#"
            ---
            table_name: my-table
            col_metadata:
              - name: one
                coltype:
                  Continuous:
                    hyper: ~
              - name: two
                coltype:
                  Categorical:
                    k: 2
              - name: two
                coltype:
                  Categorical:
                    k: 2
            "#
        );
        let _cb: Codebook = serde_yaml::from_str(&raw).unwrap();
    }

    #[test]
    fn serialize_metadata_list() {
        let codebook = Codebook {
            table_name: "my-table".into(),
            state_alpha_prior: None,
            view_alpha_prior: None,
            comments: None,
            row_names: None,
            col_metadata: vec![
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: "one".into(),
                    notes: None,
                    coltype: ColType::Continuous { hyper: None },
                },
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: "two".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        value_map: None,
                    },
                },
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: "three".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                },
            ]
            .try_into()
            .unwrap(),
        };

        let cb_string = serde_yaml::to_string(&codebook).unwrap();

        let raw = indoc!(
            r#"
            ---
            table_name: my-table
            col_metadata:
              - name: one
                coltype:
                  Continuous: {}
              - name: two
                coltype:
                  Categorical:
                    k: 2
              - name: three
                coltype:
                  Categorical:
                    k: 3"#
        );

        assert_eq!(cb_string, raw)
    }

    #[test]
    fn serialize_then_deserialize() {
        let codebook = Codebook {
            table_name: "my-table".into(),
            state_alpha_prior: None,
            view_alpha_prior: None,
            comments: None,
            row_names: None,
            col_metadata: vec![
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: "one".into(),
                    notes: None,
                    coltype: ColType::Continuous { hyper: None },
                },
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: "two".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        value_map: None,
                    },
                },
                ColMetadata {
                    spec_type: SpecType::Other,
                    name: "three".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                },
            ]
            .try_into()
            .unwrap(),
        };

        let cb_string = serde_yaml::to_string(&codebook).unwrap();
        let new_codebook: Codebook = serde_yaml::from_str(&cb_string).unwrap();

        assert!(new_codebook == codebook);
    }
}

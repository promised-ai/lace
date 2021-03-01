use std::collections::{BTreeMap, HashMap};
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use super::error::{InsertRowError, MergeColumnsError};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::ng::NgHyper;
use braid_stats::prior::pg::PgHyper;
use rv::dist::{Gamma, Kumaraswamy, NormalInvChiSquared, SymmetricDirichlet};
use serde::{Deserialize, Serialize};

/// A structure that enforces unique IDs and row names.
///
/// # Notes
///
/// Serializes to a `Vec` of `String` and deserializes to a `Vec` of
/// `String`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(into = "Vec<String>")]
#[serde(try_from = "Vec<String>")]
pub struct RowNameList {
    row_names: Vec<String>,
    index_lookup: HashMap<String, usize>,
}

impl TryFrom<Vec<String>> for RowNameList {
    type Error = String;

    fn try_from(row_names: Vec<String>) -> Result<Self, Self::Error> {
        let index_lookup: HashMap<String, usize> = row_names
            .iter()
            .enumerate()
            .map(|(ix, row_name)| (row_name.clone(), ix))
            .collect();

        if index_lookup.len() != row_names.len() {
            Err(String::from("Duplicate row names"))
        } else {
            Ok(RowNameList {
                row_names,
                index_lookup,
            })
        }
    }
}

impl Into<Vec<String>> for RowNameList {
    fn into(self) -> Vec<String> {
        self.row_names
    }
}

impl std::ops::Index<usize> for RowNameList {
    type Output = String;

    fn index(&self, ix: usize) -> &Self::Output {
        &self.row_names[ix]
    }
}

impl RowNameList {
    pub fn new() -> RowNameList {
        RowNameList {
            row_names: Vec::new(),
            index_lookup: HashMap::new(),
        }
    }

    pub fn from_range(range: std::ops::Range<usize>) -> RowNameList {
        let mut row_names: Vec<String> = Vec::new();
        let index_lookup: HashMap<String, usize> = range
            .map(|ix| {
                let row_name = format!("{}", ix);
                row_names.push(row_name.clone());
                (row_name, ix)
            })
            .collect();

        RowNameList {
            row_names,
            index_lookup,
        }
    }

    pub fn len(&self) -> usize {
        self.row_names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_names.is_empty()
    }

    pub fn index(&self, row_name: &str) -> Option<usize> {
        self.index_lookup.get(row_name).cloned()
    }

    pub fn name(&self, ix: usize) -> &String {
        &self.row_names[ix]
    }

    pub fn insert(&mut self, row_name: String) -> Result<(), InsertRowError> {
        use std::collections::hash_map::Entry;

        let ix = self.len();
        match self.index_lookup.entry(row_name.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(ix);
                Ok(row_name)
            }
            _ => Err(InsertRowError(row_name)),
        }
        .map(|row_name| self.row_names.push(row_name))
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<String, usize> {
        self.index_lookup.iter()
    }

    pub fn remove(&mut self, row_name: &str) -> bool {
        if let Some(ix) = self.index_lookup.remove(row_name) {
            self.row_names.remove(ix);
            self.index_lookup.values_mut().for_each(|val| {
                if *val > ix {
                    *val -= 1;
                }
            });
            true
        } else {
            false
        }
    }

    pub fn pop_front(&mut self) -> String {
        let row_name = self.row_names.remove(0);
        let _lookup = self.index_lookup.remove(&row_name);
        // make sure the indices align properly
        self.index_lookup.values_mut().for_each(|val| {
            *val -= 1;
        });
        row_name
    }
}

impl Default for RowNameList {
    fn default() -> RowNameList {
        RowNameList::new()
    }
}

/// A structure that enforces unique IDs and names.
///
/// #Notes
/// Serializes to a `Vec` of `ColMetadata` and deserializes to a `Vec` of
/// `ColMetadata`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(into = "Vec<ColMetadata>", try_from = "Vec<ColMetadata>")]
pub struct ColMetadataList {
    metadata: Vec<ColMetadata>,
    index_lookup: HashMap<String, usize>,
}

impl ColMetadataList {
    /// Create a new `ColMetadataList`. Returns an error -- the column name --
    /// if any of the `ColMetadata`s' are not unique (case sensitive).
    pub fn new(metadata: Vec<ColMetadata>) -> Result<Self, String> {
        let mut index_lookup = HashMap::new();
        metadata
            .iter()
            .enumerate()
            .try_for_each(|(ix, md)| {
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

    /// Append a new column to the end of the list. Returns an error if the
    /// column's name already exists.
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
            _ => Err(md.name),
        }
    }

    /// Iterate through the column metadata
    pub fn iter(&self) -> std::slice::Iter<ColMetadata> {
        self.metadata.iter()
    }

    /// The number of columns
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// True if there are no columns
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// True if one of the columns has `name`
    pub fn contains_key(&self, name: &str) -> bool {
        self.index_lookup.contains_key(name)
    }

    /// Return the integer index and the metadata of the column with `name` if
    /// it exists. Otherwise return `None`.
    pub fn get(&self, name: &str) -> Option<(usize, &ColMetadata)> {
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

impl std::ops::IndexMut<usize> for ColMetadataList {
    fn index_mut(&mut self, ix: usize) -> &mut ColMetadata {
        &mut self.metadata[ix]
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

impl TryFrom<Vec<ColMetadata>> for ColMetadataList {
    type Error = String;

    fn try_from(mds: Vec<ColMetadata>) -> Result<ColMetadataList, Self::Error> {
        ColMetadataList::new(mds)
            .map_err(|col| format!("Duplicate column name: '{}'", col))
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
    pub col_metadata: ColMetadataList,
    /// Optional misc comments
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub comments: Option<String>,
    /// Names of each row
    #[serde(skip_serializing_if = "RowNameList::is_empty")]
    #[serde(default)]
    pub row_names: RowNameList,
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
            row_names: RowNameList::new(),
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
        self.append_col_metadata(other.col_metadata)
    }

    /// Add the columns of the other codebook into this codebook. Returns a
    /// map, indexed by the old column IDs, containing the new IDs.
    pub fn append_col_metadata(
        &mut self,
        col_metadata: ColMetadataList,
    ) -> Result<(), MergeColumnsError> {
        let mut new_col_metadata: Vec<_> = col_metadata.into();
        for colmd in new_col_metadata.drain(..) {
            self.col_metadata
                .push(colmd)
                .map_err(MergeColumnsError::DuplicateColumnName)?;
        }
        Ok(())
    }
}

/// Stores metadata for the specific column types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub enum ColType {
    /// Univariate continuous (Gaussian) data model
    Continuous {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hyper: Option<NgHyper>,
        /// The normal gamma prior on components in this column. If set, the
        /// hyper prior will be ignored and the prior parameters will not be
        /// updated during inference.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<NormalInvChiSquared>,
    },
    /// Categorical data up to 256 instances
    Categorical {
        /// The number of values this column can take on. For example, if values
        /// in the column are binary, k would be 2.
        k: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hyper: Option<CsdHyper>,
        /// A Map of values from integer codes to string values. Example: 0 ->
        /// Female, 1 -> Male.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value_map: Option<BTreeMap<usize, String>>,
        /// The normal gamma prior on components in this column. If set, the
        /// hyper prior will be ignored and the prior parameters will not be
        /// updated during inference.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<SymmetricDirichlet>,
    },
    /// Discrete count-type data in [0,  ∞)
    Count {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hyper: Option<PgHyper>,
        /// The normal gamma prior on components in this column. If set, the
        /// hyper prior will be ignored and the prior parameters will not be
        /// updated during inference.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<Gamma>,
    },
    /// Human-labeled categorical data
    Labeler {
        n_labels: u8,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pr_h: Option<Kumaraswamy>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pr_k: Option<Kumaraswamy>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
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

    pub fn is_count(&self) -> bool {
        match self {
            ColType::Count { .. } => true,
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

    /// Return true if the prior is set, in which case the hyper prior should be
    /// ignored, and the prior parameters should not be updated.
    pub fn ignore_hyper(&self) -> bool {
        match self {
            ColType::Continuous { prior, .. } => prior.is_some(),
            ColType::Categorical { prior, .. } => prior.is_some(),
            ColType::Count { prior, .. } => prior.is_some(),
            ColType::Labeler { .. } => false,
        }
    }
}

/// The metadata associated with a column. In addition to the id and name, it
/// contains information about the data model of a column.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ColMetadata {
    /// The name of the Column
    pub name: String,
    /// The column model
    pub coltype: ColType,
    /// Optional notes about the column
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use std::convert::TryInto;

    fn quick_codebook() -> Codebook {
        let coltype = ColType::Categorical {
            k: 2,
            hyper: None,
            prior: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
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
            prior: None,
            value_map: None,
        };
        let md0 = ColMetadata {
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md1 = ColMetadata {
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
        };
        let md2 = ColMetadata {
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
                prior: None,
            };
            let md0 = ColMetadata {
                name: "fwee".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let md1 = ColMetadata {
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
                prior: None,
            };
            let md0 = ColMetadata {
                name: "1".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let md1 = ColMetadata {
                name: "four".to_string(),
                coltype: coltype.clone(),
                notes: None,
            };
            let col_metadata = ColMetadataList::new(vec![md0, md1]).unwrap();
            Codebook::new("table2".to_string(), col_metadata)
        };

        match cb1.merge_cols(cb2) {
            Err(MergeColumnsError::DuplicateColumnName(col)) => {
                assert_eq!(col, String::from("1"))
            }
            Ok(_) => panic!("merge should have detected duplicate"),
        }
    }

    #[test]
    fn lookup_for_continuous_coltype_is_none() {
        let coltype = ColType::Continuous {
            hyper: None,
            prior: None,
        };
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
            prior: None,
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
            prior: None,
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
            row_names: RowNameList::new(),
            col_metadata: ColMetadataList::try_from(vec![
                ColMetadata {
                    name: "one".into(),
                    notes: None,
                    coltype: ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                },
                ColMetadata {
                    name: "two".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        prior: None,
                        value_map: None,
                    },
                },
                ColMetadata {
                    name: "three".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        prior: None,
                        value_map: None,
                    },
                },
            ])
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
            row_names: RowNameList::new(),
            col_metadata: ColMetadataList::try_from(vec![
                ColMetadata {
                    name: "one".into(),
                    notes: None,
                    coltype: ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                },
                ColMetadata {
                    name: "two".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        prior: None,
                        value_map: None,
                    },
                },
                ColMetadata {
                    name: "three".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        prior: None,
                        value_map: None,
                    },
                },
            ])
            .unwrap(),
        };

        let cb_string = serde_yaml::to_string(&codebook).unwrap();
        let new_codebook: Codebook = serde_yaml::from_str(&cb_string).unwrap();

        assert!(new_codebook == codebook);
    }

    #[test]
    fn row_names_try_into_unique_vec() {
        let names: Vec<String> = vec![
            String::from("one"),
            String::from("two"),
            String::from("three"),
            String::from("four"),
            String::from("five"),
        ];

        let row_names: RowNameList = names.try_into().unwrap();

        assert_eq!(row_names.len(), 5);
        assert_eq!(row_names.index("one"), Some(0));
        assert_eq!(row_names.index("two"), Some(1));
        assert_eq!(row_names.index("three"), Some(2));
        assert_eq!(row_names.index("four"), Some(3));
        assert_eq!(row_names.index("five"), Some(4));
    }

    #[test]
    fn row_names_try_into_repeats_vec() {
        let names: Vec<String> = vec![
            String::from("one"),
            String::from("two"),
            String::from("three"),
            String::from("three"),
            String::from("five"),
        ];

        let res: Result<RowNameList, String> = names.try_into();
        assert!(res.is_err());
    }

    #[test]
    fn insert_into_empty_row_names() {
        let mut row_names = RowNameList::new();

        assert!(row_names.is_empty());

        assert!(row_names.insert(String::from("one")).is_ok());
        assert_eq!(row_names.index("one"), Some(0));
        assert_eq!(row_names.len(), 1);

        assert!(row_names.insert(String::from("two")).is_ok());
        assert_eq!(row_names.index("two"), Some(1));
        assert_eq!(row_names.len(), 2);
    }

    #[test]
    fn insert_existing_row_names_returns_error() {
        let mut row_names = RowNameList::new();
        assert!(row_names.insert(String::from("one")).is_ok());

        let res = row_names.insert(String::from("one"));
        match res {
            Err(InsertRowError(name)) => {
                assert_eq!(name, String::from("one"));
            }
            _ => panic!("should have been InsertRowError"),
        }
    }

    #[test]
    fn pop_front() {
        let mut row_names = RowNameList::new();
        assert!(row_names.insert(String::from("one")).is_ok());
        assert!(row_names.insert(String::from("two")).is_ok());
        assert!(row_names.insert(String::from("three")).is_ok());

        assert_eq!(row_names.len(), 3);

        assert_eq!(row_names.pop_front(), String::from("one"));

        assert_eq!(row_names.len(), 2);
        assert_eq!(row_names[0], "two");
        assert_eq!(row_names[1], "three");
    }
}

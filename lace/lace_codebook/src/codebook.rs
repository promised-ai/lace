use crate::data::df_to_codebook;
use crate::error::{
    CodebookError, ColMetadataListError, InsertRowError, MergeColumnsError,
    RowNameListError,
};
use crate::ValueMap;
#[cfg(feature = "experimental")]
use lace_stats::experimental::sbd::SbdHyper;
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::rv::dist::{Gamma, NormalInvChiSquared, SymmetricDirichlet};
#[cfg(feature = "experimental")]
use lace_stats::rv::experimental::Sb;
use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[cfg(feature = "experimental")]
use lace_stats::experimental::dp_discrete::DpdPrior;

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
    type Error = RowNameListError;

    fn try_from(row_names: Vec<String>) -> Result<Self, Self::Error> {
        let mut index_lookup: HashMap<String, usize> = HashMap::new();
        row_names
            .iter()
            .enumerate()
            .try_for_each(|(ix, row_name)| {
                if let Some(old_ix) = index_lookup.insert(row_name.clone(), ix)
                {
                    Err(RowNameListError::Duplicate {
                        row_name: row_name.clone(),
                        ix_1: old_ix,
                        ix_2: ix,
                    })
                } else {
                    Ok(())
                }
            })?;

        Ok(RowNameList {
            row_names,
            index_lookup,
        })
    }
}

impl From<RowNameList> for Vec<String> {
    fn from(rows: RowNameList) -> Self {
        rows.row_names
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

    pub fn with_capacity(n: usize) -> RowNameList {
        RowNameList {
            row_names: Vec::with_capacity(n),
            index_lookup: HashMap::with_capacity(n),
        }
    }

    pub fn from_range(range: std::ops::Range<usize>) -> RowNameList {
        let mut row_names: Vec<String> = Vec::new();
        let index_lookup: HashMap<String, usize> = range
            .map(|ix| {
                let row_name = format!("{ix}");
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

    pub fn name(&self, ix: usize) -> Option<&String> {
        if ix >= self.row_names.len() {
            None
        } else {
            Some(&self.row_names[ix])
        }
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

    /// Return the last row name. Returns `None` if the list is empty
    pub fn last(&mut self) -> Option<&String> {
        self.row_names.last()
    }

    pub fn as_slice(&self) -> &[String] {
        self.row_names.as_slice()
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
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(into = "Vec<ColMetadata>", try_from = "Vec<ColMetadata>")]
pub struct ColMetadataList {
    metadata: Vec<ColMetadata>,
    index_lookup: HashMap<String, usize>,
}

impl ColMetadataList {
    // TODO: new should return an empty list. This constructor should be
    // from_vec
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

    /// Get the name of the column at index ix if it exists
    pub fn name(&self, ix: usize) -> Option<&String> {
        if ix >= self.metadata.len() {
            None
        } else {
            Some(&self.metadata[ix].name)
        }
    }

    /// Return the integer index and the metadata of the column with `name` if
    /// it exists. Otherwise return `None`.
    pub fn get(&self, name: &str) -> Option<(usize, &ColMetadata)> {
        self.index_lookup
            .get(name)
            .map(|&ix| (ix, &self.metadata[ix]))
    }

    /// Take the column metadata with given key
    pub fn take(&mut self, name: &str) -> Option<ColMetadata> {
        let ix_opt = self.index_lookup.remove(name);
        if let Some(ix) = ix_opt {
            self.index_lookup.iter_mut().for_each(|(_, i)| {
                if *i > ix {
                    *i -= 1;
                }
            });
            Some(self.metadata.remove(ix))
        } else {
            None
        }
    }

    /// Remove the entries at `ix` and re-index
    pub fn remove_by_index(&mut self, ix: usize) {
        let removed = self.metadata.remove(ix);

        self.index_lookup
            .remove(removed.name.as_str())
            .expect("column not in list");

        for (i, colmd) in self.metadata.iter().enumerate().skip(ix) {
            *self.index_lookup.get_mut(colmd.name.as_str()).unwrap() = i;
        }
    }
}

impl From<ColMetadataList> for Vec<ColMetadata> {
    fn from(cols: ColMetadataList) -> Self {
        cols.metadata
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

impl std::ops::Index<&str> for ColMetadataList {
    type Output = ColMetadata;

    fn index(&self, name: &str) -> &Self::Output {
        let ix = self.index_lookup[name];
        &self.metadata[ix]
    }
}

impl std::ops::IndexMut<&str> for ColMetadataList {
    fn index_mut(&mut self, name: &str) -> &mut ColMetadata {
        let ix = self.index_lookup[name];
        &mut self.metadata[ix]
    }
}

impl TryFrom<Vec<ColMetadata>> for ColMetadataList {
    type Error = ColMetadataListError;

    fn try_from(mds: Vec<ColMetadata>) -> Result<ColMetadataList, Self::Error> {
        ColMetadataList::new(mds).map_err(ColMetadataListError::Duplicate)
    }
}

/// Codebook object for storing information about the dataset
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Codebook {
    /// The name of the table
    pub table_name: String,
    /// Prior on State CRP alpha parameter
    pub state_alpha_prior: Option<Gamma>,
    /// Prior on View CRP alpha parameters
    pub view_alpha_prior: Option<Gamma>,
    /// The metadata for each column indexed by name
    pub col_metadata: ColMetadataList,
    /// Optional misc comments
    pub comments: Option<String>,
    /// Names of each row
    pub row_names: RowNameList,
}

impl Default for Codebook {
    fn default() -> Codebook {
        Codebook::new(String::from("my_table"), ColMetadataList::default())
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

    /// Create a codebook from a polars DataFrame
    ///
    /// # Arguments
    /// - df: the dataframe
    /// - cat_cutoff: the maximum value an integer column can take on before it
    ///   is considered Count type instead of Categorical
    /// - alpha_prior_opt: Optional Gamma prior on the column and row CRP alpha
    /// - no_hypers: if `true` do not do prior parameter inference
    pub fn from_df(
        df: &DataFrame,
        cat_cutoff: Option<u8>,
        alpha_prior_opt: Option<Gamma>,
        no_hypers: bool,
    ) -> Result<Self, CodebookError> {
        df_to_codebook(df, cat_cutoff, alpha_prior_opt, no_hypers)
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
    pub fn n_cols(&self) -> usize {
        self.col_metadata.len()
    }

    /// Get the number of rows
    pub fn n_rows(&self) -> usize {
        self.row_names.len()
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

    /// Get the integer index of a row by name
    pub fn row_index(&self, row_name: &str) -> Option<usize> {
        self.row_names.index(row_name)
    }

    /// Get the integer index of a column by name
    pub fn column_index(&self, col_name: &str) -> Option<usize> {
        self.col_metadata.get(col_name).map(|(ix, _)| ix)
    }

    /// Return the ValueMap of the column if it exists
    ///
    /// Will return `None` if the column is not categorical
    pub fn value_map(&self, col_ix: usize) -> Option<&ValueMap> {
        self.col_metadata[col_ix].coltype.value_map()
    }
}

// TODO: snake case variants
/// Stores metadata for the specific column types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub enum ColType {
    /// Univariate continuous (Gaussian) data model
    Continuous {
        hyper: Option<NixHyper>,
        /// The normal inverse chi-squared prior on components in this column.
        /// If set, the hyper prior will be ignored and the prior parameters
        /// will not be updated during inference.
        prior: Option<NormalInvChiSquared>,
    },
    /// Categorical data up to 256 instances
    Categorical {
        /// The number of values this column can take on. For example, if values
        /// in the column are binary, k would be 2.
        k: usize,
        hyper: Option<CsdHyper>,
        /// Store the category type and a map from usize index to category (and
        /// the reverse).
        value_map: ValueMap,
        /// The normal gamma prior on components in this column. If set, the
        /// hyper prior will be ignored and the prior parameters will not be
        /// updated during inference.
        prior: Option<SymmetricDirichlet>,
    },
    /// Discrete count-type data in [0,  ∞)
    Count {
        hyper: Option<PgHyper>,
        /// The normal gamma prior on components in this column. If set, the
        /// hyper prior will be ignored and the prior parameters will not be
        /// updated during inference.
        prior: Option<Gamma>,
    },
    #[cfg(feature = "experimental")]
    /// Index type
    Index {
        hyper: Option<SbdHyper>,
        prior: Option<Sb>,
    },
}

impl ColType {
    pub fn is_continuous(&self) -> bool {
        matches!(self, ColType::Continuous { .. })
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self, ColType::Categorical { .. })
    }

    pub fn is_count(&self) -> bool {
        matches!(self, ColType::Count { .. })
    }

    /// Return the value map if the type is categorical and a value map exists.
    pub fn value_map(&self) -> Option<&ValueMap> {
        match self {
            ColType::Categorical { value_map, .. } => Some(value_map),
            _ => None,
        }
    }

    /// Return true if the prior is set, in which case the hyper prior should be
    /// ignored, and the prior parameters should not be updated.
    pub fn ignore_hyper(&self) -> bool {
        match self {
            ColType::Continuous { prior, .. } => prior.is_some(),
            ColType::Categorical { prior, .. } => prior.is_some(),
            ColType::Count { prior, .. } => prior.is_some(),
            #[cfg(feature = "experimental")]
            ColType::Index { prior, .. } => prior.is_some(),
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
    pub notes: Option<String>,
    /// True if missing data should be treated as random
    #[serde(default)]
    pub missing_not_at_random: bool,
    // True if this is a latent column
    #[serde(default)]
    pub latent: bool,
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
            value_map: ValueMap::U8(2),
        };
        let md0 = ColMetadata {
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
            missing_not_at_random: false,
            latent: false,
        };
        let md1 = ColMetadata {
            name: "1".to_string(),
            coltype: coltype.clone(),
            notes: None,
            missing_not_at_random: false,
            latent: false,
        };
        let md2 = ColMetadata {
            name: "2".to_string(),
            coltype,
            notes: None,
            missing_not_at_random: false,
            latent: false,
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
            value_map: ValueMap::U8(2),
        };
        let md0 = ColMetadata {
            name: "0".to_string(),
            coltype: coltype.clone(),
            notes: None,
            missing_not_at_random: false,
            latent: false,
        };
        let md1 = ColMetadata {
            name: "2".to_string(),
            coltype: coltype.clone(),
            notes: None,
            missing_not_at_random: false,
            latent: false,
        };
        let md2 = ColMetadata {
            name: "2".to_string(),
            coltype,
            notes: None,
            missing_not_at_random: false,
            latent: false,
        };

        let col_metadata = ColMetadataList::new(vec![md0, md1, md2]);

        assert_eq!(col_metadata, Err(String::from("2")));
    }

    #[test]
    fn n_cols_returns_number_of_column_metadata() {
        let cb = quick_codebook();
        assert_eq!(cb.n_cols(), 3);
    }

    #[test]
    fn merge_codebooks_produces_correct_ids() {
        let mut cb1 = quick_codebook();
        let cb2 = {
            let coltype = ColType::Categorical {
                k: 2,
                hyper: None,
                value_map: ValueMap::U8(2),
                prior: None,
            };
            let md0 = ColMetadata {
                name: "fwee".to_string(),
                coltype: coltype.clone(),
                notes: None,
                missing_not_at_random: false,
                latent: false,
            };
            let md1 = ColMetadata {
                name: "four".to_string(),
                coltype,
                notes: None,
                missing_not_at_random: false,
                latent: false,
            };
            let col_metadata = ColMetadataList::new(vec![md0, md1]).unwrap();
            Codebook::new("table2".to_string(), col_metadata)
        };

        cb1.merge_cols(cb2).unwrap();
        assert_eq!(cb1.n_cols(), 5);

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
                value_map: ValueMap::U8(2),
                prior: None,
            };
            let md0 = ColMetadata {
                name: "1".to_string(),
                coltype: coltype.clone(),
                notes: None,
                missing_not_at_random: false,
                latent: false,
            };
            let md1 = ColMetadata {
                name: "four".to_string(),
                coltype,
                notes: None,
                missing_not_at_random: false,
                latent: false,
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
    fn value_map_for_continuous_coltype_is_none() {
        let coltype = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        assert!(coltype.value_map().is_none());
    }

    #[test]
    fn value_map_for_categorical_coltype_check() {
        use std::collections::BTreeSet;
        let mut cats: BTreeSet<String> = BTreeSet::new();
        cats.insert("dog".into());
        cats.insert("cat".into());
        cats.insert("hamster".into());
        let coltype = ColType::Categorical {
            k: 3,
            hyper: None,
            prior: None,
            value_map: ValueMap::new(cats),
        };
        if let Some(value_map) = coltype.value_map() {
            assert_eq!(value_map.len(), 3);
            assert_eq!(value_map.ix(&("cat").into()), Some(0_usize));
            assert_eq!(value_map.ix(&("dog").into()), Some(1_usize));
            assert_eq!(value_map.ix(&("hamster").into()), Some(2_usize));
            assert_eq!(value_map.ix(&("gerbil").into()), None);
        } else {
            panic!("Failed")
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
                  !Continuous
                    hyper: ~
              - name: two
                coltype:
                  !Categorical
                    k: 2
                    value_map: !u8 2
              - name: three
                coltype:
                  !Categorical
                    k: 2
                    value_map: !u8 2
            state_alpha_prior: ~
            view_alpha_prior: ~
            comments: ~
            row_names:
                - one
                - two
                - three
            "#
        );
        let cb: Codebook = serde_yaml::from_str(raw).unwrap();
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
                  !Continuous
                    hyper: ~
              - name: two
                coltype:
                  !Categorical
                    k: 2
              - name: two
                coltype:
                  !Categorical
                    k: 2
            state_alpha_prior: ~
            view_alpha_prior: ~
            comments: ~
            row_names:
                - one
                - two
                - three
            "#
        );
        let _cb: Codebook = serde_yaml::from_str(raw).unwrap();
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
                    missing_not_at_random: false,
                    latent: false,
                },
                ColMetadata {
                    name: "two".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        prior: None,
                        value_map: ValueMap::U8(2),
                    },
                    missing_not_at_random: false,
                    latent: false,
                },
                ColMetadata {
                    name: "three".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        prior: None,
                        value_map: ValueMap::U8(3),
                    },
                    missing_not_at_random: false,
                    latent: false,
                },
            ])
            .unwrap(),
        };

        let cb_string = serde_yaml::to_string(&codebook).unwrap();
        let raw = indoc!(
            r#"
            table_name: my-table
            state_alpha_prior: null
            view_alpha_prior: null
            col_metadata:
            - name: one
              coltype: !Continuous
                hyper: null
                prior: null
              notes: null
              missing_not_at_random: false
              latent: false
            - name: two
              coltype: !Categorical
                k: 2
                hyper: null
                value_map: !u8 2
                prior: null
              notes: null
              missing_not_at_random: false
              latent: false
            - name: three
              coltype: !Categorical
                k: 3
                hyper: null
                value_map: !u8 3
                prior: null
              notes: null
              missing_not_at_random: false
              latent: false
            comments: null
            row_names: []
            "#
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
                    missing_not_at_random: false,
                    latent: false,
                },
                ColMetadata {
                    name: "two".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 2,
                        hyper: None,
                        prior: None,
                        value_map: ValueMap::U8(2),
                    },
                    missing_not_at_random: false,
                    latent: false,
                },
                ColMetadata {
                    name: "three".into(),
                    notes: None,
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        prior: None,
                        value_map: ValueMap::U8(3),
                    },
                    missing_not_at_random: false,
                    latent: false,
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

        let res: Result<RowNameList, RowNameListError> = names.try_into();
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

    mod colmetedatalist {
        use super::*;

        fn get_colmds(n: usize) -> ColMetadataList {
            let mut colmds = ColMetadataList::default();
            for i in 0..n {
                let colmd = ColMetadata {
                    name: format!("{}", i),
                    notes: None,
                    coltype: ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                    missing_not_at_random: false,
                    latent: false,
                };
                colmds.push(colmd).unwrap();
            }
            colmds
        }

        #[test]
        fn remove_by_index_from_front() {
            let mut colmds = get_colmds(5);

            assert_eq!(colmds.len(), 5);
            assert_eq!(colmds[0].name, String::from("0"));
            assert_eq!(colmds.get("0").unwrap().0, 0);
            assert_eq!(colmds.get("0").unwrap().1.name, String::from("0"));

            colmds.remove_by_index(0);

            assert_eq!(colmds.len(), 4);
            assert_eq!(colmds[0].name, String::from("1"));
            assert_eq!(colmds[1].name, String::from("2"));
            assert_eq!(colmds.get("0"), None);
            assert_eq!(colmds.get("1").unwrap().0, 0);
            assert_eq!(colmds.get("1").unwrap().1.name, String::from("1"));
        }

        #[test]
        fn remove_by_index_from_middle() {
            let mut colmds = get_colmds(5);

            assert_eq!(colmds.len(), 5);
            assert_eq!(colmds[2].name, String::from("2"));
            assert_eq!(colmds.get("2").unwrap().0, 2);
            assert_eq!(colmds.get("2").unwrap().1.name, String::from("2"));

            colmds.remove_by_index(2);

            assert_eq!(colmds.len(), 4);

            assert_eq!(colmds[0].name, String::from("0"));
            assert_eq!(colmds[1].name, String::from("1"));
            assert_eq!(colmds[2].name, String::from("3"));
            assert_eq!(colmds[3].name, String::from("4"));

            assert_eq!(colmds.get("0").unwrap().0, 0);
            assert_eq!(colmds.get("1").unwrap().0, 1);
            assert_eq!(colmds.get("3").unwrap().0, 2);
            assert_eq!(colmds.get("4").unwrap().0, 3);

            assert_eq!(colmds.get("0").unwrap().1.name, String::from("0"));
            assert_eq!(colmds.get("1").unwrap().1.name, String::from("1"));
            assert_eq!(colmds.get("3").unwrap().1.name, String::from("3"));
            assert_eq!(colmds.get("4").unwrap().1.name, String::from("4"));
        }
    }
}

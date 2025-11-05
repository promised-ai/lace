use serde::Deserialize;
use serde::Serialize;

use crate::codebook::Codebook;
use crate::error::IndexError;

/// Trait defining an item that can be converted into a row index
pub trait RowIndex: Clone + std::fmt::Debug {
    /// Use the codebook to return the integer row index
    ///
    /// # Example
    /// ```
    /// use lace::examples::Example;
    /// use lace::RowIndex;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// // The first row
    /// let ix = "antelope".row_ix(&oracle.codebook).unwrap();
    /// assert_eq!(ix, 0);
    ///
    /// // "flys" is a column name
    /// let ix_res = "flys".row_ix(&oracle.codebook);
    /// assert!(ix_res.is_err());
    /// ```
    fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError>;

    /// Return the item as a string reference, if possible
    ///
    /// # Example
    /// ```
    /// use lace::RowIndex;
    ///
    /// let ix = String::from("antelope");
    /// assert!(ix.row_str().is_some());
    ///
    /// let ix = 10_usize;
    /// assert!(ix.row_str().is_none());
    /// ```
    fn row_str(&self) -> Option<&str>;

    /// Return the item as a usize, if possible
    ///
    /// # Example
    /// ```
    /// use lace::RowIndex;
    ///
    /// let ix = String::from("antelope");
    /// assert!(ix.row_usize().is_none());
    ///
    /// let ix = 10_usize;
    /// assert_eq!(ix.row_usize(), Some(10));
    /// ```
    fn row_usize(&self) -> Option<usize>;
}

impl RowIndex for usize {
    fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        let n_rows = codebook.row_names.len();
        if *self < n_rows {
            Ok(*self)
        } else {
            Err(IndexError::RowIndexOutOfBounds {
                n_rows,
                row_ix: *self,
            })
        }
    }

    fn row_str(&self) -> Option<&str> {
        None
    }

    fn row_usize(&self) -> Option<usize> {
        Some(*self)
    }
}

impl RowIndex for String {
    fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        codebook.row_index(self.as_str()).ok_or_else(|| {
            IndexError::RowNameDoesNotExist { name: self.clone() }
        })
    }

    fn row_str(&self) -> Option<&str> {
        Some(self.as_str())
    }

    fn row_usize(&self) -> Option<usize> {
        None
    }
}

impl<'a> RowIndex for &'a str {
    fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        codebook.row_index(self).ok_or_else(|| {
            IndexError::RowNameDoesNotExist {
                name: String::from(*self),
            }
        })
    }

    fn row_str(&self) -> Option<&str> {
        Some(self)
    }

    fn row_usize(&self) -> Option<usize> {
        None
    }
}

/// Trait defining items that can converted into a usize column index
pub trait ColumnIndex: Clone + std::fmt::Debug {
    /// Use the codebook to return the integer column index
    ///
    /// # Example
    /// ```
    /// use lace::examples::Example;
    /// use lace::ColumnIndex;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// // "flys" is the 35th column (index 34)
    /// let ix = "flys".col_ix(&oracle.codebook).unwrap();
    /// assert_eq!(ix, 34);
    ///
    /// // "antelope" os a row
    /// let ix_res = "antelope".col_ix(&oracle.codebook);
    /// assert!(ix_res.is_err());
    /// ```
    fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError>;

    /// Return the item as a string reference, if possible
    ///
    /// # Example
    /// ```
    /// use lace::ColumnIndex;
    ///
    /// let ix = String::from("flys");
    /// assert!(ix.col_str().is_some());
    ///
    /// let ix = 10_usize;
    /// assert!(ix.col_str().is_none());
    /// ```
    fn col_str(&self) -> Option<&str>;

    /// Return the item as a usize, if possible
    ///
    /// # Example
    /// ```
    /// use lace::ColumnIndex;
    ///
    /// let ix = String::from("flys");
    /// assert!(ix.col_usize().is_none());
    ///
    /// let ix = 10_usize;
    /// assert_eq!(ix.col_usize(), Some(10));
    /// ```
    fn col_usize(&self) -> Option<usize>;
}

impl ColumnIndex for usize {
    fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        let n_cols = codebook.n_cols();
        if *self < n_cols {
            Ok(*self)
        } else {
            Err(IndexError::ColumnIndexOutOfBounds {
                n_cols,
                col_ix: *self,
            })
        }
    }

    fn col_str(&self) -> Option<&str> {
        None
    }

    fn col_usize(&self) -> Option<usize> {
        Some(*self)
    }
}

impl ColumnIndex for String {
    fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        codebook.column_index(self.as_str()).ok_or_else(|| {
            IndexError::ColumnNameDoesNotExist { name: self.clone() }
        })
    }

    fn col_str(&self) -> Option<&str> {
        Some(self.as_str())
    }

    fn col_usize(&self) -> Option<usize> {
        None
    }
}

impl<'a> ColumnIndex for &'a str {
    fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        codebook.column_index(self).ok_or_else(|| {
            IndexError::ColumnNameDoesNotExist {
                name: String::from(*self),
            }
        })
    }

    fn col_str(&self) -> Option<&str> {
        Some(self)
    }

    fn col_usize(&self) -> Option<usize> {
        None
    }
}

/// Holds a `String` name or a `usize` index
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(untagged, rename_all = "snake_case")]
pub enum NameOrIndex {
    Name(String),
    Index(usize),
}

macro_rules! impl_name_or_index {
    ($ty: ty) => {
        impl RowIndex for $ty {
            fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
                match self {
                    NameOrIndex::Name(name) => name.row_ix(codebook),
                    NameOrIndex::Index(ix) => ix.row_ix(codebook),
                }
            }

            fn row_str(&self) -> Option<&str> {
                match self {
                    NameOrIndex::Name(name) => Some(name.as_str()),
                    NameOrIndex::Index(_) => None,
                }
            }

            fn row_usize(&self) -> Option<usize> {
                match self {
                    NameOrIndex::Name(_) => None,
                    NameOrIndex::Index(ix) => Some(*ix),
                }
            }
        }

        impl ColumnIndex for $ty {
            fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
                match self {
                    NameOrIndex::Name(name) => name.col_ix(codebook),
                    NameOrIndex::Index(ix) => ix.col_ix(codebook),
                }
            }

            fn col_str(&self) -> Option<&str> {
                match self {
                    NameOrIndex::Name(name) => Some(name.as_str()),
                    NameOrIndex::Index(_) => None,
                }
            }

            fn col_usize(&self) -> Option<usize> {
                match self {
                    NameOrIndex::Name(_) => None,
                    NameOrIndex::Index(ix) => Some(*ix),
                }
            }
        }
    };
}

impl_name_or_index!(NameOrIndex);
impl_name_or_index!(&NameOrIndex);

impl From<usize> for NameOrIndex {
    fn from(ix: usize) -> Self {
        NameOrIndex::Index(ix)
    }
}

impl From<String> for NameOrIndex {
    fn from(name: String) -> Self {
        NameOrIndex::Name(name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableIndex<R: RowIndex, C: ColumnIndex> {
    /// Represents an entire row
    Row(R),
    /// Represents an entire column
    Column(C),
    /// Represents a single cell
    Cell(R, C),
}

impl<R: RowIndex, C: ColumnIndex> From<(R, C)> for TableIndex<R, C> {
    fn from(value: (R, C)) -> Self {
        TableIndex::Cell(value.0, value.1)
    }
}

impl<R: RowIndex, C: ColumnIndex> TableIndex<R, C> {
    #[inline]
    pub fn is_row(&self) -> bool {
        matches!(self, TableIndex::Row(_))
    }

    #[inline]
    pub fn is_column(&self) -> bool {
        matches!(self, TableIndex::Column(_))
    }

    #[inline]
    pub fn is_cell(&self) -> bool {
        matches!(self, TableIndex::Cell(..))
    }

    /// Returns `true` if this index is in the codebook
    pub fn in_codebook(&self, codebook: &Codebook) -> bool {
        match &self {
            TableIndex::Row(row_ix) => row_ix.row_ix(codebook).is_ok(),
            TableIndex::Column(col_ix) => col_ix.col_ix(codebook).is_ok(),
            TableIndex::Cell(row_ix, col_ix) => {
                let row_in = row_ix.row_ix(codebook).is_ok();
                let col_in = col_ix.col_ix(codebook).is_ok();
                row_in && col_in
            }
        }
    }

    /// Convert an index to an integer type index.
    #[inline]
    pub fn into_usize_index(
        self,
        codebook: &Codebook,
    ) -> Result<TableIndex<usize, usize>, IndexError> {
        match self {
            TableIndex::Row(row_ix) => {
                row_ix.row_ix(codebook).map(TableIndex::Row)
            }
            TableIndex::Column(col_ix) => {
                col_ix.col_ix(codebook).map(TableIndex::Column)
            }
            TableIndex::Cell(row_ix, col_ix) => {
                row_ix.row_ix(codebook).and_then(|rix| {
                    col_ix
                        .col_ix(codebook)
                        .map(|cix| TableIndex::Cell(rix, cix))
                })
            }
        }
    }
}

pub(crate) fn extract_colixs<Ix: ColumnIndex>(
    col_ixs: &[Ix],
    codebook: &Codebook,
) -> Result<Vec<usize>, IndexError> {
    col_ixs
        .iter()
        .map(|col_ix| col_ix.col_ix(codebook))
        .collect()
}

pub(crate) fn extract_col_pair<Ix: ColumnIndex>(
    pair: &(Ix, Ix),
    codebook: &Codebook,
) -> Result<(usize, usize), IndexError> {
    pair.0
        .col_ix(codebook)
        .and_then(|ix_a| pair.1.col_ix(codebook).map(|ix_b| (ix_a, ix_b)))
}

pub(crate) fn extract_row_pair<Ix: RowIndex>(
    pair: &(Ix, Ix),
    codebook: &Codebook,
) -> Result<(usize, usize), IndexError> {
    pair.0
        .row_ix(codebook)
        .and_then(|ix_a| pair.1.row_ix(codebook).map(|ix_b| (ix_a, ix_b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_cell() {
        let cell = TableIndex::Cell(0, "one");
        let serialized = serde_json::to_string(&cell).unwrap();
        let target = String::from(r#"{"cell":[0,"one"]}"#);
        println!("{}", serialized);
        assert_eq!(serialized, target);
    }

    #[test]
    fn json_row() {
        let row = TableIndex::<usize, usize>::Row(0_usize);
        let serialized = serde_json::to_string(&row).unwrap();
        let target = String::from(r#"{"row":0}"#);
        println!("{}", serialized);
        assert_eq!(serialized, target);
    }

    #[test]
    fn json_column() {
        let col = TableIndex::<usize, usize>::Column(0);
        let serialized = serde_json::to_string(&col).unwrap();
        let target = String::from(r#"{"column":0}"#);
        println!("{}", serialized);
        assert_eq!(serialized, target);
    }

    #[test]
    fn to_cell() {
        let cell: TableIndex<_, _> = (0, "one").into();
        let target = TableIndex::Cell(0, "one");
        assert_eq!(cell, target);
    }
}

use braid_codebook::Codebook;
use serde::{Deserialize, Serialize};

use crate::error::IndexError;

pub trait RowIndex {
    fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError>;
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
}

impl RowIndex for String {
    fn row_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        codebook.row_index(self.as_str()).ok_or_else(|| {
            IndexError::RowNameDoesNotExist { name: self.clone() }
        })
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
}

pub trait ColumnIndex {
    fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError>;
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
}

impl ColumnIndex for String {
    fn col_ix(&self, codebook: &Codebook) -> Result<usize, IndexError> {
        codebook.column_index(self.as_str()).ok_or_else(|| {
            IndexError::ColumnNameDoesNotExist { name: self.clone() }
        })
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
}

// /// Holds a `String` name or a `usize` index
// #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
// #[serde(untagged, rename_all = "snake_case")]
// pub enum NameOrIndex {
//     Name(String),
//     Index(usize),
// }

// impl NameOrIndex {
//     /// Returns a reference to the name if this is a `Name` variant
//     pub fn name(&self) -> Option<&str> {
//         match self {
//             Self::Name(s) => Some(s.as_str()),
//             _ => None,
//         }
//     }

//     /// Returns a `usize` index if this is an `Index` variant
//     pub fn index(&self) -> Option<usize> {
//         match self {
//             Self::Index(ix) => Some(*ix),
//             _ => None,
//         }
//     }
// }

// impl From<usize> for NameOrIndex {
//     fn from(ix: usize) -> Self {
//         Self::Index(ix)
//     }
// }

// impl From<&usize> for NameOrIndex {
//     fn from(ix: &usize) -> Self {
//         Self::Index(*ix)
//     }
// }

// impl From<&str> for NameOrIndex {
//     fn from(name: &str) -> Self {
//         Self::Name(String::from(name))
//     }
// }

// impl From<String> for NameOrIndex {
//     fn from(name: String) -> Self {
//         Self::Name(name)
//     }
// }

// impl From<&String> for NameOrIndex {
//     fn from(name: &String) -> Self {
//         Self::Name(name.clone())
//     }
// }

// /// A row index
// #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
// #[serde(rename_all = "snake_case")]
// pub struct RowIndex(pub NameOrIndex);

// impl RowIndex {
//     #[inline]
//     pub(crate) fn into_index_if_in_codebook(
//         self,
//         codebook: &Codebook,
//     ) -> Result<Self, usize> {
//         get_row_index(self, codebook)
//     }
// }

// impl<T: Into<NameOrIndex>> From<T> for RowIndex {
//     fn from(t: T) -> Self {
//         Self(t.into())
//     }
// }

// /// A column index
// #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
// #[serde(rename_all = "snake_case")]
// pub struct ColumnIndex(pub NameOrIndex);

// impl ColumnIndex {
//     #[inline]
//     pub(crate) fn into_index_if_in_codebook(
//         self,
//         codebook: &Codebook,
//     ) -> Result<Self, usize> {
//         get_column_index(self, codebook)
//     }
// }

// impl<T: Into<NameOrIndex>> From<T> for ColumnIndex {
//     fn from(t: T) -> Self {
//         Self(t.into())
//     }
// }

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

// impl From<RowIndex> for TableIndex {
//     fn from(ix: RowIndex) -> Self {
//         Self::Row(ix)
//     }
// }
//
// impl From<ColumnIndex> for TableIndex {
//     fn from(ix: ColumnIndex) -> Self {
//         Self::Column(ix)
//     }
// }
//
// impl<R, C> From<(R, C)> for TableIndex
// where
//     R: Into<RowIndex>,
//     C: Into<ColumnIndex>,
// {
//     fn from(ixs: (R, C)) -> Self {
//         Self::Cell(ixs.0.into(), ixs.1.into())
//     }
// }

// /// TODO: more consistency between row and column metadata lists
// #[inline]
// fn row_in_codebook(row_ix: &RowIndex, codebook: &Codebook) -> bool {
//     match &row_ix.0 {
//         NameOrIndex::Name(name) => {
//             codebook.row_names.index(name.as_str()).is_some()
//         }
//         NameOrIndex::Index(ix) => *ix >= codebook.row_names.len(),
//     }
// }

// #[inline]
// fn col_in_codebook(col_ix: &ColumnIndex, codebook: &Codebook) -> bool {
//     match &col_ix.0 {
//         NameOrIndex::Name(name) => {
//             codebook.col_metadata.get(name.as_str()).is_some()
//         }
//         NameOrIndex::Index(ix) => *ix >= codebook.col_metadata.len(),
//     }
// }

// fn get_row_index(
//     index: RowIndex,
//     codebook: &Codebook,
// ) -> Result<RowIndex, usize> {
//     match &index.0 {
//         NameOrIndex::Name(name) => Ok(codebook
//             .row_index(name)
//             .map(|ix| RowIndex(NameOrIndex::Index(ix)))
//             .unwrap_or(index)),
//         NameOrIndex::Index(ix) => {
//             if *ix < codebook.row_names.len() {
//                 Ok(index)
//             } else {
//                 Err(*ix)
//             }
//         }
//     }
// }

// fn get_column_index(
//     index: ColumnIndex,
//     codebook: &Codebook,
// ) -> Result<ColumnIndex, usize> {
//     match &index.0 {
//         NameOrIndex::Name(name) => Ok(codebook
//             .col_metadata
//             .get(name)
//             .map(|(ix, _)| ColumnIndex(NameOrIndex::Index(ix)))
//             .unwrap_or(index)),
//         NameOrIndex::Index(ix) => {
//             if *ix < codebook.col_metadata.len() {
//                 Ok(index)
//             } else {
//                 Err(*ix)
//             }
//         }
//     }
// }

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

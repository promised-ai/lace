use braid_codebook::Codebook;
use serde::{Deserialize, Serialize};

/// Holds a `String` name or a `usize` index
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum NameOrIndex {
    Name(String),
    Index(usize),
}

impl NameOrIndex {
    /// Returns a reference to the name if this is a `Name` variant
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Name(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns a `usize` index if this is an `Index` variant
    pub fn index(&self) -> Option<usize> {
        match self {
            Self::Index(ix) => Some(*ix),
            _ => None,
        }
    }
}

impl From<usize> for NameOrIndex {
    fn from(ix: usize) -> Self {
        Self::Index(ix)
    }
}

impl From<&str> for NameOrIndex {
    fn from(name: &str) -> Self {
        Self::Name(String::from(name))
    }
}

impl From<String> for NameOrIndex {
    fn from(name: String) -> Self {
        Self::Name(name)
    }
}

/// A row index
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RowIndex(pub NameOrIndex);

impl RowIndex {
    #[inline]
    pub(crate) fn into_index_if_in_codebook(
        self,
        codebook: &Codebook,
    ) -> Result<Self, usize> {
        get_row_index(self, codebook)
    }
}

impl<T: Into<NameOrIndex>> From<T> for RowIndex {
    fn from(t: T) -> Self {
        Self(t.into())
    }
}

/// A column index
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ColumnIndex(pub NameOrIndex);

impl ColumnIndex {
    #[inline]
    pub(crate) fn into_index_if_in_codebook(
        self,
        codebook: &Codebook,
    ) -> Result<Self, usize> {
        get_column_index(self, codebook)
    }
}

impl<T: Into<NameOrIndex>> From<T> for ColumnIndex {
    fn from(t: T) -> Self {
        Self(t.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableIndex {
    /// Represents an entire row
    Row(RowIndex),
    /// Represents an entire column
    Column(ColumnIndex),
    /// Represents a single cell
    Cell(RowIndex, ColumnIndex),
}

impl From<RowIndex> for TableIndex {
    fn from(ix: RowIndex) -> Self {
        Self::Row(ix)
    }
}

impl From<ColumnIndex> for TableIndex {
    fn from(ix: ColumnIndex) -> Self {
        Self::Column(ix)
    }
}

impl<R, C> From<(R, C)> for TableIndex
where
    R: Into<RowIndex>,
    C: Into<ColumnIndex>,
{
    fn from(ixs: (R, C)) -> Self {
        Self::Cell(ixs.0.into(), ixs.1.into())
    }
}

/// TODO: more consistency between row and column metadata lists
#[inline]
fn row_in_codebook(row_ix: &RowIndex, codebook: &Codebook) -> bool {
    match &row_ix.0 {
        NameOrIndex::Name(name) => {
            codebook.row_names.index(name.as_str()).is_some()
        }
        NameOrIndex::Index(ix) => *ix >= codebook.row_names.len(),
    }
}

#[inline]
fn col_in_codebook(col_ix: &ColumnIndex, codebook: &Codebook) -> bool {
    match &col_ix.0 {
        NameOrIndex::Name(name) => {
            codebook.col_metadata.get(name.as_str()).is_some()
        }
        NameOrIndex::Index(ix) => *ix >= codebook.col_metadata.len(),
    }
}

fn get_row_index(
    index: RowIndex,
    codebook: &Codebook,
) -> Result<RowIndex, usize> {
    match &index.0 {
        NameOrIndex::Name(name) => Ok(codebook
            .row_index(name)
            .map(|ix| RowIndex(NameOrIndex::Index(ix)))
            .unwrap_or(index)),
        NameOrIndex::Index(ix) => {
            if *ix < codebook.row_names.len() {
                Ok(index)
            } else {
                Err(*ix)
            }
        }
    }
}

fn get_column_index(
    index: ColumnIndex,
    codebook: &Codebook,
) -> Result<ColumnIndex, usize> {
    match &index.0 {
        NameOrIndex::Name(name) => Ok(codebook
            .col_metadata
            .get(name)
            .map(|(ix, _)| ColumnIndex(NameOrIndex::Index(ix)))
            .unwrap_or(index)),
        NameOrIndex::Index(ix) => {
            if *ix < codebook.col_metadata.len() {
                Ok(index)
            } else {
                Err(*ix)
            }
        }
    }
}

impl TableIndex {
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
            TableIndex::Row(row_ix) => row_in_codebook(row_ix, codebook),
            TableIndex::Column(col_ix) => col_in_codebook(col_ix, codebook),
            TableIndex::Cell(row_ix, col_ix) => {
                let row_in = row_in_codebook(row_ix, codebook);
                let col_in = col_in_codebook(col_ix, codebook);
                row_in && col_in
            }
        }
    }

    /// If this index is in the codebook, convert to and index; if it is not in
    /// the codebook, ensure that it is a name, otherwise, return None
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn into_index_if_in_codebook(
        self,
        codebook: &Codebook,
    ) -> Option<Self> {
        match self {
            Self::Row(ix) => get_row_index(ix, codebook).map(Self::Row).ok(),
            Self::Column(ix) => {
                get_column_index(ix, codebook).map(Self::Column).ok()
            }
            Self::Cell(row_ix, col_ix) => {
                let row_opt = get_row_index(row_ix, codebook).ok();
                let col_opt = get_column_index(col_ix, codebook).ok();
                if let (Some(row), Some(col)) = (row_opt, col_opt) {
                    Some(Self::Cell(row, col))
                } else {
                    None
                }
            }
        }
    }

    /// Convert an index to an integer type index.
    #[inline]
    pub fn into_usize_index(self, codebook: &Codebook) -> Option<Self> {
        match self {
            Self::Row(RowIndex(NameOrIndex::Index(_))) => Some(self),
            Self::Column(ColumnIndex(NameOrIndex::Index(_))) => Some(self),
            Self::Cell(
                RowIndex(NameOrIndex::Index(_)),
                ColumnIndex(NameOrIndex::Index(_)),
            ) => Some(self),
            Self::Row(RowIndex(NameOrIndex::Name(name))) => codebook
                .row_index(name.as_str())
                .map(|ix| RowIndex::from(ix).into()),
            Self::Column(ColumnIndex(NameOrIndex::Name(name))) => codebook
                .column_index(name.as_str())
                .map(|ix| ColumnIndex::from(ix).into()),
            Self::Cell(
                RowIndex(NameOrIndex::Name(row)),
                ColumnIndex(NameOrIndex::Name(col)),
            ) => codebook
                .row_index(row.as_str())
                .map(NameOrIndex::Index)
                .and_then(|row_ix| {
                    codebook
                        .column_index(col.as_str())
                        .map(|col_ix| (row_ix, col_ix).into())
                }),
            Self::Cell(RowIndex(NameOrIndex::Name(row)), col_ix) => codebook
                .row_index(row.as_str())
                .map(|row_ix| (row_ix, col_ix).into()),
            Self::Cell(row_ix, ColumnIndex(NameOrIndex::Name(col))) => codebook
                .column_index(col.as_str())
                .map(|col_ix| (row_ix, col_ix).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_cell() {
        let cell = TableIndex::Cell(
            RowIndex(NameOrIndex::Index(0)),
            ColumnIndex(NameOrIndex::Name("one".into())),
        );
        let serialized = serde_json::to_string(&cell).unwrap();
        let target = String::from(r#"{"cell":[0,"one"]}"#);
        println!("{}", serialized);
        assert_eq!(serialized, target);
    }

    #[test]
    fn json_row() {
        let row = TableIndex::Row(RowIndex(NameOrIndex::Index(0)));
        let serialized = serde_json::to_string(&row).unwrap();
        let target = String::from(r#"{"row":0}"#);
        println!("{}", serialized);
        assert_eq!(serialized, target);
    }

    #[test]
    fn json_column() {
        let col = TableIndex::Column(ColumnIndex(NameOrIndex::Index(0)));
        let serialized = serde_json::to_string(&col).unwrap();
        let target = String::from(r#"{"column":0}"#);
        println!("{}", serialized);
        assert_eq!(serialized, target);
    }

    #[test]
    fn to_cell() {
        let cell: TableIndex = (0, "one").into();
        let target = TableIndex::Cell(
            RowIndex(NameOrIndex::Index(0)),
            ColumnIndex(NameOrIndex::Name("one".into())),
        );
        assert_eq!(cell, target);
    }

    #[test]
    fn to_row_index() {
        {
            let row: RowIndex = 0.into();
            assert_eq!(row, RowIndex(NameOrIndex::Index(0)));
        }

        {
            let row: RowIndex = "one".into();
            assert_eq!(row, RowIndex(NameOrIndex::Name("one".into())));
        }

        {
            let ix: TableIndex = RowIndex::from(0).into();
            assert_eq!(ix, TableIndex::Row(RowIndex(NameOrIndex::Index(0))));
        }
    }

    #[test]
    fn to_column_index() {
        {
            let col: ColumnIndex = 0.into();
            assert_eq!(col, ColumnIndex(NameOrIndex::Index(0)));
        }

        {
            let col: ColumnIndex = "one".into();
            assert_eq!(col, ColumnIndex(NameOrIndex::Name("one".into())));
        }
    }
}

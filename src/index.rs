use serde::{Deserialize, Serialize};

/// Holds a `String` name or a `usize` index
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum NameOrIndex {
    Name(String),
    Index(usize),
}

impl From<usize> for NameOrIndex {
    fn from(ix: usize) -> Self {
        NameOrIndex::Index(ix)
    }
}

impl From<&str> for NameOrIndex {
    fn from(name: &str) -> Self {
        NameOrIndex::Name(String::from(name))
    }
}

impl From<String> for NameOrIndex {
    fn from(name: String) -> Self {
        NameOrIndex::Name(name)
    }
}

/// A row index
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RowIndex(NameOrIndex);

impl<T: Into<NameOrIndex>> From<T> for RowIndex {
    fn from(t: T) -> Self {
        RowIndex(t.into())
    }
}

/// A column index
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ColumnIndex(NameOrIndex);

impl<T: Into<NameOrIndex>> From<T> for ColumnIndex {
    fn from(t: T) -> Self {
        ColumnIndex(t.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
        TableIndex::Row(ix)
    }
}

impl From<ColumnIndex> for TableIndex {
    fn from(ix: ColumnIndex) -> Self {
        TableIndex::Column(ix)
    }
}

impl<R, C> From<(R, C)> for TableIndex
where
    R: Into<RowIndex>,
    C: Into<ColumnIndex>,
{
    fn from(ixs: (R, C)) -> Self {
        TableIndex::Cell(ixs.0.into(), ixs.1.into())
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

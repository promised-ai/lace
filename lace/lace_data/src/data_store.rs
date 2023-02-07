use std::collections::BTreeMap;
use std::ops::Index;

use serde::{Deserialize, Serialize};

use crate::{Container, Datum, FeatureData};

/// Stores the data for an `Oracle`
///
/// # Notes
///
/// To save space, the data is removed from `State`s when they're saved to a
/// lacefile. The `Oracle` only needs one copy of the data, so when an
/// `Oracle` is loaded, the data is kept separate to avoid loading a copy of the
/// data for each `State` in the `Oracle`.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DataStore(pub BTreeMap<usize, FeatureData>);

impl Index<usize> for DataStore {
    type Output = FeatureData;

    fn index(&self, ix: usize) -> &Self::Output {
        &self.0[&ix]
    }
}

macro_rules! data_store_get_arm {
    ($variant:ident, $xs: expr, $row_ix: expr) => {
        $xs.get($row_ix)
            .map(Datum::$variant)
            .unwrap_or(Datum::Missing)
    };
}

impl DataStore {
    pub fn new(data: BTreeMap<usize, FeatureData>) -> Self {
        DataStore(data)
    }

    /// Get the datum at [row_ix, col_ix] as a `Datum`
    pub fn get(&self, row_ix: usize, col_ix: usize) -> Datum {
        // TODO: SparseContainer index get (xs[i]) should return an option
        match self.0[&col_ix] {
            FeatureData::Binary(ref xs) => {
                data_store_get_arm!(Binary, xs, row_ix)
            }
            FeatureData::Continuous(ref xs) => {
                data_store_get_arm!(Continuous, xs, row_ix)
            }
            FeatureData::Categorical(ref xs) => {
                data_store_get_arm!(Categorical, xs, row_ix)
            }
            FeatureData::Count(ref xs) => {
                data_store_get_arm!(Count, xs, row_ix)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseContainer;

    fn fixture() -> DataStore {
        let dc1: SparseContainer<f64> = SparseContainer::from(vec![
            (4.0, true),
            (3.0, false),
            (2.0, true),
            (1.0, true),
            (0.0, true),
        ]);

        let dc2: SparseContainer<u8> = SparseContainer::from(vec![
            (5, true),
            (3, true),
            (2, true),
            (1, false),
            (4, true),
        ]);

        let mut data = BTreeMap::<usize, FeatureData>::new();
        data.insert(0, FeatureData::Continuous(dc1));
        data.insert(1, FeatureData::Categorical(dc2));
        DataStore(data)
    }

    #[test]
    fn gets_present_continuous_data() {
        let ds = fixture();
        assert_eq!(ds.get(0, 0), Datum::Continuous(4.0));
        assert_eq!(ds.get(2, 0), Datum::Continuous(2.0));
    }

    #[test]
    fn gets_present_categorical_data() {
        let ds = fixture();
        assert_eq!(ds.get(0, 1), Datum::Categorical(5));
        assert_eq!(ds.get(4, 1), Datum::Categorical(4));
    }

    #[test]
    fn gets_missing_continuous_data() {
        let ds = fixture();
        assert_eq!(ds.get(1, 0), Datum::Missing);
    }

    #[test]
    fn gets_missing_categorical_data() {
        let ds = fixture();
        assert_eq!(ds.get(3, 1), Datum::Missing);
    }
}

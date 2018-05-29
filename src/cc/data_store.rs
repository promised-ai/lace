use cc::{DType, FeatureData};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DataStore(BTreeMap<usize, FeatureData>);

impl DataStore {
    pub fn new(data: BTreeMap<usize, FeatureData>) -> Self {
        DataStore(data)
    }

    pub fn get(&self, row_ix: usize, col_ix: usize) -> DType {
        // TODO: DataContainer index get (xs[i]) should return an option
        match self.0[&col_ix] {
            FeatureData::Continuous(ref xs) => {
                if xs.present[row_ix] {
                    DType::Continuous(xs[row_ix])
                } else {
                    DType::Missing
                }
            }
            FeatureData::Categorical(ref xs) => {
                if xs.present[row_ix] {
                    DType::Categorical(xs[row_ix])
                } else {
                    DType::Missing
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc::DataContainer;

    fn fixture() -> DataStore {
        let dc1: DataContainer<f64> = DataContainer {
            data: vec![4.0, 3.0, 2.0, 1.0, 0.0],
            present: vec![true, false, true, true, true],
        };

        let dc2: DataContainer<u8> = DataContainer {
            data: vec![5, 3, 2, 1, 4],
            present: vec![true, true, true, false, true],
        };

        let mut data = BTreeMap::<usize, FeatureData>::new();
        data.insert(0, FeatureData::Continuous(dc1));
        data.insert(1, FeatureData::Categorical(dc2));
        DataStore(data)
    }

    #[test]
    fn gets_present_continuous_data() {
        let ds = fixture();
        assert_eq!(ds.get(0, 0), DType::Continuous(4.0));
        assert_eq!(ds.get(2, 0), DType::Continuous(2.0));
    }

    #[test]
    fn gets_present_categorical_data() {
        let ds = fixture();
        assert_eq!(ds.get(0, 1), DType::Categorical(5));
        assert_eq!(ds.get(4, 1), DType::Categorical(4));
    }

    #[test]
    fn gets_missing_continuous_data() {
        let ds = fixture();
        assert_eq!(ds.get(1, 0), DType::Missing);
    }

    #[test]
    fn gets_missing_categorical_data() {
        let ds = fixture();
        assert_eq!(ds.get(3, 1), DType::Missing);
    }

}

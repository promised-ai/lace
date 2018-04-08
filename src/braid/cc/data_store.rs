use std::collections::BTreeMap;
use cc::{DType, FeatureData};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DataStore(BTreeMap<usize, FeatureData>);

impl DataStore {
    pub fn new(data: BTreeMap<usize, FeatureData>) -> Self {
        DataStore(data)
    }

    pub fn get(&self, row_ix: usize, col_ix: usize) -> DType {
        match self.0[&col_ix] {
            FeatureData::Continuous(ref xs) => DType::Continuous(xs[row_ix]),
            FeatureData::Categorical(ref xs) => DType::Categorical(xs[row_ix]),
        }
    }
}

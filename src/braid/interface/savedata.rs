
use std::collections::BTreeMap;

use cc:{Codebook, DataContainer, State};
use interface::Oracle;


pub struct SaveData {
    codebook: Codebook,
    empty_states: Vec<State>,
    data: BTreeMap<usize, FeatureData>
}

// impl SaveData {
//     pub fn from_oracle(&mut oracle: Oracle) -> SaveData {
//         let codebook = oracle.codebook.clone();
//         let mut data: BTreeMap<usize, FatureData> = BTreeMap::new();
//         let first_state = &oracle.states[0];

//     }
// }

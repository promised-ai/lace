// Feature type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FType {
    Continuous,
    Categorical,
}

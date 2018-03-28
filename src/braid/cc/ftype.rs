// Feature type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FType {
    Continuous,
    Categorical,
}

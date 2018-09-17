/// Feature type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FType {
    #[serde(rename = "continuous")]
    Continuous,
    #[serde(rename = "categorical")]
    Categorical,
}

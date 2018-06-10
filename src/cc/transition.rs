#[derive(Clone, Copy, PartialEq)]
pub enum ViewTransition {
    RowAssignment,
    Alpha,
    FeaturePriors,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum StateTransition {
    #[serde(rename = "column_assignment")]
    ColumnAssignment,
    #[serde(rename = "row_assignment")]
    RowAssignment,
    #[serde(rename = "state_alpha")]
    StateAlpha,
    #[serde(rename = "view_alphas")]
    ViewAlphas,
    #[serde(rename = "feature_priors")]
    FeaturePriors,
}

impl StateTransition {
    pub fn to_view_transition(&self) -> Option<ViewTransition> {
        match self {
            StateTransition::ViewAlphas => Some(ViewTransition::Alpha),
            StateTransition::RowAssignment => {
                Some(ViewTransition::RowAssignment)
            }
            StateTransition::FeaturePriors => {
                Some(ViewTransition::FeaturePriors)
            }
            _ => None,
        }
    }

    pub fn to_view_transitions(ts: &Vec<Self>) -> Vec<ViewTransition> {
        ts.iter().filter_map(|t| t.to_view_transition()).collect()
    }
}

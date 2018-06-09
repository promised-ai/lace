#[derive(Clone, Copy, PartialEq)]
pub enum ViewTransition {
    RowAssignment,
    Alpha,
    FeaturePriors,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum StateTransition {
    ColumnAssignment,
    RowAssignment,
    StateAlpha,
    ViewAlphas,
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

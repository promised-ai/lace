use std::fmt;

#[derive(Clone, Copy, PartialEq)]
pub enum ViewTransition {
    RowAssignment,
    Alpha,
    FeaturePriors,
    ComponentParams,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum StateTransition {
    /// Reassign columns to views
    #[serde(rename = "column_assignment")]
    ColumnAssignment,
    /// Reassign rows in views to categories
    #[serde(rename = "row_assignment")]
    RowAssignment,
    /// Update the alpha (discount) parameter on the column-to-views CRP
    #[serde(rename = "state_alpha")]
    StateAlpha,
    /// Update the alpha (discount) parameters on the row-to-categories CRP
    #[serde(rename = "view_alphas")]
    ViewAlphas,
    /// Update the feature (column) prior parameters
    #[serde(rename = "feature_priors")]
    FeaturePriors,
    /// Update the parameters in the feature components. This is usually done
    /// automatically during the row assignment, but if the row assignment is
    /// not done (e.g. in the case of Geweke testing), then you can turn it on
    /// with this transition Note: this is not a default state transition.
    #[serde(rename = "component_params")]
    ComponentParams,
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
            StateTransition::ComponentParams => {
                Some(ViewTransition::ComponentParams)
            }
            _ => None,
        }
    }

    pub fn to_view_transitions(ts: &Vec<Self>) -> Vec<ViewTransition> {
        ts.iter().filter_map(|t| t.to_view_transition()).collect()
    }
}

impl fmt::Display for StateTransition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            StateTransition::ColumnAssignment => "ColumnAssignment",
            StateTransition::RowAssignment => "RowAssignment",
            StateTransition::StateAlpha => "StateAlpha",
            StateTransition::ViewAlphas => "ViewAlphas",
            StateTransition::FeaturePriors => "FeaturePriors",
            StateTransition::ComponentParams => "ComponentParams",
        };
        write!(f, "{}", s)
    }
}

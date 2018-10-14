use result;
use std::fmt;
use std::str::FromStr;

/// MCMC transitions in the `View`
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ViewTransition {
    /// Reassign rows to categories
    RowAssignment,
    /// Update the alpha (discount) parameters on the CRP
    Alpha,
    /// Update the feature (column) prior parameters
    FeaturePriors,
    /// Update the parameters in the feature components. This is usually done
    /// automatically during the row assignment, but if the row assignment is
    /// not done (e.g. in the case of Geweke testing), then you can turn it on
    /// with this transition Note: this is not a default state transition.
    ComponentParams,
}

/// MCMC transitions in the `State`
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
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

impl FromStr for StateTransition {
    type Err = result::Error;

    fn from_str(s: &str) -> result::Result<Self> {
        match s {
            "column_assignment" => Ok(StateTransition::ColumnAssignment),
            "row_assignment" => Ok(StateTransition::RowAssignment),
            "state_alpha" => Ok(StateTransition::StateAlpha),
            "view_alphas" => Ok(StateTransition::ViewAlphas),
            "feature_priors" => Ok(StateTransition::FeaturePriors),
            "component_params" => Ok(StateTransition::ComponentParams),
            _ => {
                let err_kind = result::ErrorKind::ParseError;
                let msg = "Could not parse state transition";
                Err(result::Error::new(err_kind, msg))
            }
        }
    }
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

    pub fn extract_view_transitions(ts: &Vec<Self>) -> Vec<ViewTransition> {
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

#[cfg(test)]
mod tests {
    use super::*;

    mod state_transition {
        use super::*;

        #[test]
        fn column_assignment_from_str() {
            assert_eq!(
                StateTransition::from_str("column_assignment").unwrap(),
                StateTransition::ColumnAssignment,
            );
        }

        #[test]
        fn row_assignment_from_str() {
            assert_eq!(
                StateTransition::from_str("row_assignment").unwrap(),
                StateTransition::RowAssignment,
            );
        }

        #[test]
        fn state_alpha_from_str() {
            assert_eq!(
                StateTransition::from_str("state_alpha").unwrap(),
                StateTransition::StateAlpha
            );
        }

        #[test]
        fn view_alpha_from_str() {
            assert_eq!(
                StateTransition::from_str("view_alphas").unwrap(),
                StateTransition::ViewAlphas,
            );
        }

        #[test]
        fn component_params_from_str() {
            assert_eq!(
                StateTransition::from_str("component_params").unwrap(),
                StateTransition::ComponentParams,
            );
        }

        #[test]
        fn feature_priors_from_str() {
            assert_eq!(
                StateTransition::from_str("feature_priors").unwrap(),
                StateTransition::FeaturePriors,
            );
        }
    }
}

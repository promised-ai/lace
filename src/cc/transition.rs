use std::convert::TryFrom;
use std::fmt;
use std::str::FromStr;

use crate::ParseError;
use serde::{Deserialize, Serialize};

/// MCMC transitions in the `View`
#[derive(Deserialize, Serialize, Clone, Copy, Eq, PartialEq, Debug, Hash)]
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
#[derive(Deserialize, Serialize, Clone, Copy, Eq, PartialEq, Debug, Hash)]
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
    type Err = ParseError<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "column_assignment" => Ok(StateTransition::ColumnAssignment),
            "row_assignment" => Ok(StateTransition::RowAssignment),
            "state_alpha" => Ok(StateTransition::StateAlpha),
            "view_alphas" => Ok(StateTransition::ViewAlphas),
            "feature_priors" => Ok(StateTransition::FeaturePriors),
            "component_params" => Ok(StateTransition::ComponentParams),
            _ => Err(ParseError(s.to_owned())),
        }
    }
}

impl TryFrom<&StateTransition> for ViewTransition {
    type Error = ParseError<StateTransition>;

    fn try_from(st: &StateTransition) -> Result<ViewTransition, Self::Error> {
        match st {
            StateTransition::ViewAlphas => Ok(ViewTransition::Alpha),
            StateTransition::RowAssignment => Ok(ViewTransition::RowAssignment),
            StateTransition::FeaturePriors => Ok(ViewTransition::FeaturePriors),
            StateTransition::ComponentParams => {
                Ok(ViewTransition::ComponentParams)
            }
            _ => Err(ParseError(*st)),
        }
    }
}

impl TryFrom<StateTransition> for ViewTransition {
    type Error = ParseError<StateTransition>;

    fn try_from(st: StateTransition) -> Result<ViewTransition, Self::Error> {
        ViewTransition::try_from(&st)
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

use std::convert::TryFrom;

use serde::{Deserialize, Serialize};

use crate::alg::{ColAssignAlg, RowAssignAlg};
use crate::ParseError;

pub const DEFAULT_STATE_TRANSITIONS: [StateTransition; 5] = [
    StateTransition::ColumnAssignment(ColAssignAlg::Slice),
    StateTransition::StateAlpha,
    StateTransition::RowAssignment(RowAssignAlg::Slice),
    StateTransition::ViewAlphas,
    StateTransition::FeaturePriors,
];

/// MCMC transitions in the `View`
#[derive(Deserialize, Serialize, Clone, Copy, Eq, PartialEq, Debug)]
pub enum ViewTransition {
    /// Reassign rows to categories
    RowAssignment(RowAssignAlg),
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
#[derive(Deserialize, Serialize, Clone, Copy, Eq, PartialEq, Debug)]
pub enum StateTransition {
    /// Reassign columns to views
    #[serde(rename = "column_assignment")]
    ColumnAssignment(ColAssignAlg),
    /// Reassign rows in views to categories
    #[serde(rename = "row_assignment")]
    RowAssignment(RowAssignAlg),
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

impl TryFrom<StateTransition> for ViewTransition {
    type Error = ParseError<StateTransition>;

    fn try_from(st: StateTransition) -> Result<ViewTransition, Self::Error> {
        match st {
            StateTransition::ViewAlphas => Ok(ViewTransition::Alpha),
            StateTransition::RowAssignment(alg) => {
                Ok(ViewTransition::RowAssignment(alg))
            }
            StateTransition::FeaturePriors => Ok(ViewTransition::FeaturePriors),
            StateTransition::ComponentParams => {
                Ok(ViewTransition::ComponentParams)
            }
            _ => Err(ParseError(st)),
        }
    }
}

// TODO: Conversion tests

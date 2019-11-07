use braid_stats::Datum;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::TryInto;

/// Describes a the conditions (or not) on a conditional distribution
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, PartialOrd)]
pub enum Given {
    /// The conditions in `(column_id, value)` tuples. The tuple
    /// `(11, Datum::Continuous(2.3))` indicates that we wish to condition on
    /// the value of column 11 being 2.3.
    #[serde(rename = "conditions")]
    Conditions(Vec<(usize, Datum)>),
    /// The absence of conditioning observations
    #[serde(rename = "nothing")]
    Nothing,
}

impl Given {
    /// Determine whether there are no conditions
    ///
    /// # Example
    ///
    /// ```
    /// # use braid_stats::Datum;
    /// # use braid::Given;
    /// let nothing_given = Given::Nothing;
    ///
    /// assert!(nothing_given.is_nothing());
    ///
    /// let something_given = Given::Conditions(vec![(1, Datum::Categorical(1))]);
    ///
    /// assert!(!something_given.is_nothing());
    /// ```
    pub fn is_nothing(&self) -> bool {
        match self {
            Given::Nothing => true,
            _ => false,
        }
    }

    pub fn is_conditions(&self) -> bool {
        match self {
            Given::Conditions(..) => true,
            _ => false,
        }
    }
}

impl Default for Given {
    fn default() -> Self {
        Given::Nothing
    }
}

///
///
/// # Example
///
/// ```
/// # use braid::interface::Given;
/// # use braid::interface::IntoGivenError;
/// use std::convert::TryInto;
/// use braid_stats::Datum;
///
/// let conditions_good = vec![
///     (0, Datum::Categorical(0)),
///     (1, Datum::Categorical(0)),
/// ];
///
/// let given_good: Result<Given, IntoGivenError> = conditions_good.try_into();
/// assert!(given_good.is_ok());
///
/// // duplicate indices
/// let conditions_bad = vec![
///     (0, Datum::Categorical(0)),
///     (0, Datum::Categorical(0)),
/// ];
/// let given_bad: Result<Given, IntoGivenError> = conditions_bad.try_into();
///
/// assert_eq!(
///     given_bad.unwrap_err(),
///     IntoGivenError::DuplicateConditionIndicesError
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum IntoGivenError {
    /// Tried to covert `Some(vec![])` into a Given. Use `None` instead
    EmptyConditionsError,
    /// The same column index appears more than once
    DuplicateConditionIndicesError,
}

impl<'a> TryInto<Given> for Vec<(usize, Datum)> {
    type Error = IntoGivenError;

    fn try_into(self) -> Result<Given, Self::Error> {
        if self.is_empty() {
            Ok(Given::Nothing)
        } else {
            let mut set: HashSet<usize> = HashSet::new();
            if self.iter().any(|(ix, _)| !set.insert(ix.clone())) {
                Err(IntoGivenError::DuplicateConditionIndicesError)
            } else {
                Ok(Given::Conditions(self))
            }
        }
    }
}

impl<'a> TryInto<Given> for Option<Vec<(usize, Datum)>> {
    type Error = IntoGivenError;

    fn try_into(self) -> Result<Given, Self::Error> {
        match self {
            Some(conditions) => {
                if conditions.is_empty() {
                    Err(IntoGivenError::EmptyConditionsError)
                } else {
                    conditions.try_into()
                }
            }
            None => Ok(Given::Nothing),
        }
    }
}

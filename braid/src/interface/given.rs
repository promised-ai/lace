use crate::codebook::Codebook;
use crate::error::IndexError;
use crate::index::ColumnIndex;
use crate::Datum;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::TryInto;
use std::hash::Hash;

/// Describes a the conditions (or not) on a conditional distribution
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, PartialOrd, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Given<Ix: ColumnIndex> {
    /// The conditions in `(column_id, value)` tuples. The tuple
    /// `(11, Datum::Continuous(2.3))` indicates that we wish to condition on
    /// the value of column 11 being 2.3.
    Conditions(Vec<(Ix, Datum)>),
    /// The absence of conditioning observations
    Nothing,
}

impl<Ix: ColumnIndex> Given<Ix> {
    /// Determine whether there are no conditions
    ///
    /// # Example
    ///
    /// ```
    /// # use braid_data::Datum;
    /// # use braid::Given;
    /// let nothing_given = Given::<usize>::Nothing;
    ///
    /// assert!(nothing_given.is_nothing());
    ///
    /// let something_given = Given::Conditions(vec![(1, Datum::Categorical(1))]);
    ///
    /// assert!(!something_given.is_nothing());
    /// ```
    pub fn is_nothing(&self) -> bool {
        matches!(self, Given::Nothing)
    }

    /// Determine whether there are conditions
    ///
    /// # Example
    ///
    /// ```
    /// # use braid_data::Datum;
    /// # use braid::Given;
    /// let nothing_given = Given::<usize>::Nothing;
    ///
    /// assert!(!nothing_given.is_conditions());
    ///
    /// let something_given = Given::Conditions(vec![(1, Datum::Categorical(1))]);
    ///
    /// assert!(something_given.is_conditions());
    /// ```
    pub fn is_conditions(&self) -> bool {
        matches!(self, Given::Conditions(..))
    }

    /// Attempt to convert all indices in the condition into integers.
    ///
    /// # Notes
    ///
    /// Will return `IndexError` if any of the names do not exists or indices
    /// are out of bounds.
    pub fn canonical(
        self,
        codebook: &Codebook,
    ) -> Result<Given<usize>, IndexError> {
        match self {
            Self::Nothing => Ok(Given::Nothing),
            Self::Conditions(mut conditions) => {
                let conditions = conditions
                    .drain(..)
                    .map(|(col_ix, value)| {
                        col_ix.col_ix(codebook).map(|ix| (ix, value))
                    })
                    .collect::<Result<Vec<(usize, Datum)>, IndexError>>()?;
                Ok(Given::Conditions(conditions))
            }
        }
    }
}

impl<Ix: ColumnIndex> Default for Given<Ix> {
    fn default() -> Self {
        Self::Nothing
    }
}

///
///
/// # Example
///
/// ```
/// # use braid::Given;
/// # use braid::error::IntoGivenError;
/// use std::convert::TryInto;
/// use braid_data::Datum;
///
/// let conditions_good = vec![
///     (0_usize, Datum::Categorical(0)),
///     (1_usize, Datum::Categorical(0)),
/// ];
///
/// let given_good: Result<Given<usize>, IntoGivenError> = conditions_good.try_into();
/// assert!(given_good.is_ok());
///
/// // duplicate indices
/// let conditions_bad = vec![
///     (0_usize, Datum::Categorical(0)),
///     (0_usize, Datum::Categorical(0)),
/// ];
/// let given_bad: Result<Given<usize>, IntoGivenError> = conditions_bad.try_into();
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

impl<Ix: ColumnIndex + Hash + Eq> TryInto<Given<Ix>> for Vec<(Ix, Datum)> {
    type Error = IntoGivenError;

    fn try_into(mut self) -> Result<Given<Ix>, Self::Error> {
        if self.is_empty() {
            Ok(Given::Nothing)
        } else {
            let mut set: HashSet<Ix> = HashSet::new();
            if self.drain(..).any(|(ix, _)| !set.insert(ix)) {
                Err(IntoGivenError::DuplicateConditionIndicesError)
            } else {
                Ok(Given::Conditions(self))
            }
        }
    }
}

impl<Ix: ColumnIndex + Hash + Eq> TryInto<Given<Ix>>
    for Option<Vec<(Ix, Datum)>>
{
    type Error = IntoGivenError;

    fn try_into(self) -> Result<Given<Ix>, Self::Error> {
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

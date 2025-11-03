use std::collections::HashSet;

use crate::cc::state::State;
use crate::data::Datum;

use crate::error::{GivenError, LogpError};
use crate::Given;

// Given a set of target indices on which to condition, determine whether
// any of the target columns are conditioned upon.
//
// A column should not be both a target and a condition.
fn given_target_conflict(
    targets: &[usize],
    given: &Given<usize>,
) -> Option<usize> {
    match given {
        Given::Conditions(conditions) => {
            let ixs: HashSet<usize> =
                conditions.iter().map(|(ix, _)| *ix).collect();
            targets.iter().find(|ix| ixs.contains(ix)).cloned()
        }
        Given::Nothing => None,
    }
}

// Detect whether any of the `Datum`s are incompatible with the column ftypes
fn invalid_datum_types(
    state: &State,
    given: &Given<usize>,
) -> Result<(), GivenError> {
    match given {
        Given::Conditions(conditions) => {
            conditions.iter().try_for_each(|(col_ix, datum)| {
                let ftype = state.ftype(*col_ix);
                let ftype_compat = ftype.datum_compatible(datum);

                if datum.is_missing() && state.feature(*col_ix).not_mnar() {
                    Err(GivenError::MissingDatum { col_ix: *col_ix })
                } else if !ftype_compat.0 {
                    Err(GivenError::InvalidDatumForColumn {
                        col_ix: *col_ix,
                        ftype_req: ftype_compat.1.ftype_req,
                        ftype: ftype_compat.1.ftype,
                    })
                } else {
                    Ok(())
                }
            })
        }
        Given::Nothing => Ok(()),
    }
}

/// Finds errors in the simulate `Given`
pub fn find_given_errors(
    targets: &[usize],
    state: &State,
    given: &Given<usize>,
) -> Result<(), GivenError> {
    let n_cols = state.n_cols();
    match given {
        Given::Conditions(conditions) => {
            conditions.iter().try_for_each(|(col_ix, _)| {
                if *col_ix < n_cols {
                    Ok(())
                } else {
                    Err(GivenError::IndexError(
                        crate::error::IndexError::ColumnIndexOutOfBounds {
                            n_cols,
                            col_ix: *col_ix,
                        },
                    ))
                }
            })
        }
        Given::Nothing => Ok(()),
    }?;
    match given_target_conflict(targets, given) {
        Some(col_ix) => Err(GivenError::ColumnIndexAppearsInTarget { col_ix }),
        None => Ok(()),
    }?;

    invalid_datum_types(state, given)
}

/// Identify missing not at random column models
pub(crate) trait Mnar {
    /// True if the column is missing not at random
    fn is_mnar(&self) -> bool;

    /// False if the column is missing not at random
    fn not_mnar(&self) -> bool;
}

impl Mnar for crate::cc::feature::ColModel {
    fn is_mnar(&self) -> bool {
        matches!(self, Self::MissingNotAtRandom(_))
    }

    fn not_mnar(&self) -> bool {
        !self.is_mnar()
    }
}

/// Determine whether the values vector is ill-sized or if there are any
/// incompatible `Datum`s
pub fn find_value_conflicts(
    targets: &[usize],
    vals: &[Vec<Datum>],
    state: &State,
) -> Result<(), LogpError> {
    let ntargets = targets.len();
    vals.iter().try_for_each(|row| {
        if row.len() != ntargets {
            Err(LogpError::TargetsIndicesAndValuesMismatch {
                ntargets,
                nvals: row.len(),
            })
        } else {
            Ok(())
        }
    })?;

    vals.iter().try_for_each(|row| {
        targets
            .iter()
            .zip(row.iter())
            .try_for_each(|(&col_ix, datum)| {
                // given indices should have been validated first
                let ftype = state.ftype(col_ix);
                let ftype_compat = ftype.datum_compatible(datum);

                if datum.is_missing() && state.feature(col_ix).not_mnar() {
                    Err(LogpError::RequestedLogpOfMissing { col_ix })
                } else if !ftype_compat.0 {
                    Err(LogpError::InvalidDatumForColumn {
                        col_ix,
                        ftype_req: ftype_compat.1.ftype_req,
                        ftype: ftype_compat.1.ftype,
                    })
                } else {
                    Ok(())
                }
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::feature::FType;
    use crate::codebook::Codebook;
    use crate::data::DataStore;
    use crate::error::IndexError;
    use crate::interface::oracle::utils::load_states;
    use crate::interface::{HasStates, Oracle};
    use std::path::Path;

    fn oracle_from_yaml<P: AsRef<Path>>(filenames: Vec<P>) -> Oracle {
        let states = load_states(filenames);
        let data = DataStore::new(states[0].clone_data());
        Oracle {
            states,
            codebook: Codebook::default(),
            data,
        }
    }

    fn get_entropy_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/entropy/entropy-state-1.yaml",
            "resources/test/entropy/entropy-state-2.yaml",
        ];
        oracle_from_yaml(filenames)
    }

    #[test]
    fn given_nothing_is_ok() {
        let oracle = get_entropy_oracle_from_yaml();
        let nothing = Given::Nothing;
        assert!(find_given_errors(&[0, 1, 2], &oracle.states()[0], &nothing)
            .is_ok());
    }

    #[test]
    fn good_conditions_no_missing_ok() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions = Given::Conditions(vec![
            (1, Datum::Continuous(1.2)),
            (3, Datum::Categorical(0_u32.into())),
        ]);
        assert!(find_given_errors(&[0, 2], &oracle.states()[0], &conditions)
            .is_ok());
    }

    #[test]
    fn target_conflict_bad() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions =
            Given::Conditions(vec![(1, Datum::Missing), (2, Datum::Missing)]);
        let res = find_given_errors(&[0, 2], &oracle.states()[0], &conditions);
        let err = GivenError::ColumnIndexAppearsInTarget { col_ix: 2 };
        assert_eq!(res.unwrap_err(), err);
    }

    #[test]
    fn incompatible_datum_bad() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions = Given::Conditions(vec![
            (1, Datum::Continuous(1.1)),
            (3, Datum::Continuous(1.2)),
        ]);
        let res = find_given_errors(&[0, 2], &oracle.states()[0], &conditions);
        let err = GivenError::InvalidDatumForColumn {
            col_ix: 3,
            ftype_req: FType::Continuous,
            ftype: FType::Categorical,
        };
        assert_eq!(res.unwrap_err(), err);
    }

    #[test]
    fn target_index_oob_bad() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions = Given::Conditions(vec![
            (1, Datum::Continuous(1.2)),
            (4, Datum::Categorical(0_u32.into())),
        ]);
        let res = find_given_errors(&[0, 2], &oracle.states()[0], &conditions);
        let err = GivenError::IndexError(IndexError::ColumnIndexOutOfBounds {
            n_cols: 4,
            col_ix: 4,
        });
        assert_eq!(res.unwrap_err(), err);
    }
}

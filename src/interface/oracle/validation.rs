use crate::cc::State;
use crate::error::{GivenError, LogpError};
use crate::Given;
use braid_stats::Datum;
use std::collections::HashSet;

// Given a set of target indices on which to condition, determine whether
// any of the target columns are conditioned upon.
//
// A column should not be both a target and a condition.
fn given_target_conflict(targets: &[usize], given: &Given) -> Option<usize> {
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
fn invalid_datum_types(state: &State, given: &Given) -> Option<usize> {
    match given {
        Given::Conditions(conditions) => {
            conditions
                .iter()
                .find(|(col_ix, datum)| {
                    // given indices should have been validated first
                    let ftype = state.ftype(*col_ix);
                    !ftype.datum_compatible(datum)
                })
                .map(|(col_ix, _)| *col_ix)
        }
        Given::Nothing => None,
    }
}

// Check if any of the column indices in the given are out of bounds
fn column_indidices_oob(ncols: usize, given: &Given) -> bool {
    match given {
        Given::Conditions(conditions) => {
            conditions.iter().any(|(col_ix, _)| *col_ix >= ncols)
        }
        Given::Nothing => false,
    }
}

/// Finds errors in the simulate `Given`
pub fn find_given_errors(
    targets: &[usize],
    state: &State,
    given: &Given,
) -> Result<(), GivenError> {
    if column_indidices_oob(state.ncols(), given) {
        Err(GivenError::ColumnIndexOutOfBoundsError)
    } else {
        match given_target_conflict(targets, given) {
            Some(col_ix) => {
                Err(GivenError::ColumnIndexAppearsInTargetError { col_ix })
            }
            None => Ok(()),
        }?;

        match invalid_datum_types(state, given) {
            Some(col_ix) => {
                Err(GivenError::InvalidDatumForColumnError { col_ix })
            }
            None => Ok(()),
        }
    }
}

/// Determine whether the values vector is ill-sized or if there are any
/// incompatible `Datum`s
pub fn find_value_conflicts(
    targets: &[usize],
    vals: &[Vec<Datum>],
    state: &State,
) -> Result<(), LogpError> {
    let n_targets = targets.len();
    if vals.iter().any(|row| row.len() != n_targets) {
        return Err(LogpError::TargetsIndicesAndValuesMismatchError);
    }
    vals.iter()
        .map(|row| {
            let res: Result<(), LogpError> = targets
                .iter()
                .zip(row.iter())
                .map(|(col_ix, datum)| {
                    // given indices should have been validated first
                    let ftype = state.ftype(*col_ix);
                    if ftype.datum_compatible(datum) {
                        Ok(())
                    } else {
                        Err(LogpError::InvalidDatumForColumnError {
                            col_ix: *col_ix,
                        })
                    }
                })
                .collect();
            res
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::DataStore;
    use crate::interface::utils::load_states;
    use crate::interface::{HasStates, Oracle};
    use braid_codebook::Codebook;
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
            (3, Datum::Categorical(0)),
        ]);
        assert!(find_given_errors(&[0, 2], &oracle.states()[0], &conditions)
            .is_ok());
    }

    #[test]
    fn good_conditions_all_missing_ok() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions =
            Given::Conditions(vec![(1, Datum::Missing), (3, Datum::Missing)]);
        assert!(find_given_errors(&[0, 2], &oracle.states()[0], &conditions)
            .is_ok());
    }

    #[test]
    fn target_conflict_bad() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions =
            Given::Conditions(vec![(1, Datum::Missing), (2, Datum::Missing)]);
        let res = find_given_errors(&[0, 2], &oracle.states()[0], &conditions);
        let err = GivenError::ColumnIndexAppearsInTargetError { col_ix: 2 };
        assert_eq!(res.unwrap_err(), err);
    }

    #[test]
    fn incompatible_datum_bad() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions = Given::Conditions(vec![
            (1, Datum::Missing),
            (3, Datum::Continuous(1.2)),
        ]);
        let res = find_given_errors(&[0, 2], &oracle.states()[0], &conditions);
        let err = GivenError::InvalidDatumForColumnError { col_ix: 3 };
        assert_eq!(res.unwrap_err(), err);
    }

    #[test]
    fn target_index_oob_bad() {
        let oracle = get_entropy_oracle_from_yaml();
        let conditions = Given::Conditions(vec![
            (1, Datum::Continuous(1.2)),
            (4, Datum::Categorical(0)),
        ]);
        let res = find_given_errors(&[0, 2], &oracle.states()[0], &conditions);
        let err = GivenError::ColumnIndexOutOfBoundsError;
        assert_eq!(res.unwrap_err(), err);
    }
}

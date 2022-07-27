use crate::api::obj::{Datum, Given};
use crate::result::{Error, UserError};
use braid::{Engine, FType, OracleT};

/// Describes the type of dimension
#[derive(Clone)]
pub enum Dim {
    Columns,
    Rows,
}

/// Checks that someone hasn't asked for 0 samples of something
pub fn validate_n_samples(n: usize) -> Result<(), UserError> {
    if n == 0 {
        Err(UserError::from("no samples"))
    } else {
        Ok(())
    }
}

/// Checks that the columns in a `wrt` (with respect to) `rowsim` argument are
/// all in bounds (less than `n_cols`)
pub fn validate_wrt(
    wrt_opt: Option<&[usize]>,
    n_cols: usize,
) -> Result<(), Error> {
    match wrt_opt {
        Some(cols) => validate_ixs(cols, n_cols, Dim::Columns),
        None => Ok(()),
    }
}

pub fn validate_coords(
    oracle: &Engine,
    coords: &[(usize, usize)],
) -> Result<(), Error> {
    let n_rows = oracle.n_rows();
    let n_cols = oracle.n_cols();
    for (row, col) in coords.iter() {
        if *row >= n_rows || *col >= n_cols {
            let error = format!(
               "Coordinate ({}, {}) is out-of-bounds for oracle of size ({}, {})",
               row, col, n_rows, n_cols
            );
            return Err(Error::User(UserError::from(error)));
        }
    }
    Ok(())
}

/// Checks that the `ix` is in bounds
///
/// # Arguments
///
/// - ix: the index
/// - size: the number of entries along this dimension
/// - dim: the dimension
pub fn validate_ix(ix: usize, size: usize, dim: Dim) -> Result<(), Error> {
    if ix >= size {
        let error = match dim {
            Dim::Columns => {
                format!("ix is {} but there are {} columns", ix, size)
            }
            Dim::Rows => format!("ix is {} but there are {} rows", ix, size),
        };
        Err(Error::User(UserError::from(error)))
    } else {
        Ok(())
    }
}

/// Checks that the indices in `ixs` are in bounds
///
/// # Arguments
///
/// - ixs: the indices
/// - size: the number of entries along this dimension
/// - dim: the dimension
pub fn validate_ixs(ixs: &[usize], size: usize, dim: Dim) -> Result<(), Error> {
    for ix in ixs {
        validate_ix(*ix, size, dim.clone())?
    }
    Ok(())
}

/// Validates a `given` argument like those to `simulate`, `logp`, and `predict`
///
/// # Arguments
/// - oracle: The oracle to which the conditional apply
/// - targets: the indices of the columns on which we wish to impose conditions.
///   For example, the columns we are simulating, or computing probabilities
///   over.
/// - given: The conditions
pub fn validate_given(
    oracle: &Engine,
    targets: &[usize],
    given: &Given,
) -> Result<(), Error> {
    //TODO: check for repeat col_ixs
    match &given {
        Given::Conditions(conditions) => {
            if conditions.iter().any(|cond| targets.contains(&cond.0)) {
                let error = String::from(
                    "A column cannot be both a target and condition",
                );
                return Err(Error::User(UserError::from(error)));
            }
            let n_cols = oracle.n_cols();
            for (col_ix, datum) in conditions {
                if *col_ix >= n_cols {
                    let error = format!(
                        "ix is {} but there are {} columns",
                        col_ix, n_cols
                    );
                    return Err(Error::User(UserError::from(error)));
                }
                validate_datum(oracle, *col_ix, &datum.clone())?;
            }
            Ok(())
        }
        Given::Nothing => Ok(()),
    }
}

pub fn validate_logp_values(
    oracle: &Engine,
    col_ixs: &[usize],
    values: &[Vec<Datum>],
) -> Result<(), Error> {
    let ftypes_res: Result<Vec<FType>, _> = col_ixs
        .iter()
        .map(|col_ix| oracle.ftype(*col_ix).map_err(UserError::from_error))
        .collect();

    let ftypes = ftypes_res?;

    for vals in values.iter() {
        for (ftype, datum) in ftypes.iter().zip(vals.iter()) {
            let valid: bool = match ftype {
                FType::Categorical => datum.is_categorical(),
                FType::Continuous => datum.is_continuous(),
                FType::Count => datum.is_count(),
                FType::Labeler => panic!("Labeler type not supported"),
            };
            if !valid {
                let error =
                    String::from("Datum does not match the type of its column");
                return Err(Error::User(UserError::from(error)));
            }
        }
    }
    Ok(())
}

macro_rules! validate_datum_arm {
    ($col_ix:ident, $name:expr, $is_valid:expr) => {{
        if $is_valid {
            Ok(())
        } else {
            let error = format!("expected {} datum in col {}", $name, $col_ix);
            Err(UserError::from(error))
        }
    }};
}

/// Checks that `datum` is the proper type for the column at `col_ix`
pub fn validate_datum(
    oracle: &Engine,
    col_ix: usize,
    datum: &Datum,
) -> Result<(), UserError> {
    // Any OoB error should get caught before here so, unwrap should be Ok
    match oracle.ftype(col_ix).unwrap() {
        FType::Continuous => {
            validate_datum_arm!(col_ix, "continuous", datum.is_continuous())
        }
        FType::Categorical => {
            validate_datum_arm!(col_ix, "categorical", datum.is_categorical())
        }
        FType::Labeler => {
            panic!("Labeler not supported")
        }
        FType::Count => {
            validate_datum_arm!(col_ix, "count", datum.is_count())
        }
    }
}

use cc::{ColModel, DType};
use interface::Given;
use interface::error::OracleError;
use interface::oracle::Oracle;
use std::io::Result;

#[derive(Clone)]
pub enum Dim {
    Columns,
    Rows,
}

pub fn validate_n_samples(n: usize) -> Result<()> {
    if n == 0 {
        Err(OracleError::ZeroSamples.to_error())
    } else {
        Ok(())
    }
}

pub fn validate_wrt(wrt_opt: &Option<&Vec<usize>>, ncols: usize) -> Result<()> {
    match wrt_opt {
        Some(cols) => validate_ixs(cols, ncols, Dim::Columns),
        None => Ok(()),
    }
}

pub fn validate_ix(ix: usize, size: usize, dim: Dim) -> Result<()> {
    if ix >= size {
        let err = match dim {
            Dim::Columns => OracleError::ColumnIndexOutOfBounds {
                col_ix: ix,
                ncols: size,
            },
            Dim::Rows => OracleError::RowIndexOutOfBounds {
                row_ix: ix,
                nrows: size,
            },
        };
        Err(err.to_error())
    } else {
        Ok(())
    }
}

pub fn validate_ixs(ixs: &Vec<usize>, size: usize, dim: Dim) -> Result<()> {
    for ix in ixs {
        let result = validate_ix(*ix, size, dim.clone());
        if result.is_err() {
            return result;
        }
    }
    Ok(())
}

pub fn validate_given(oracle: &Oracle, given_opt: &Given) -> Result<()> {
    //TODO: check for repeat col_ixs
    //TODO: make sure col_ixs nad given col_ixs don't overlap at all
    match &given_opt {
        Some(given) => {
            let ncols = oracle.ncols();
            for (col_ix, dtype) in given {
                if *col_ix >= ncols {
                    let err = OracleError::ColumnIndexOutOfBounds {
                        col_ix: *col_ix,
                        ncols: ncols,
                    };
                    return Err(err.to_error());
                }
                validate_dtype(&oracle, *col_ix, &dtype)?;
            }
            Ok(())
        }
        None => Ok(()),
    }
}

pub fn validate_dtype(
    oracle: &Oracle,
    col_ix: usize,
    dtype: &DType,
) -> Result<()> {
    let state = &oracle.states[0];
    let view_ix = state.asgn.asgn[col_ix];
    let ftr = &state.views[view_ix].ftrs.get(&col_ix).unwrap();
    // TODO: can we kill this with macros?
    match ftr {
        ColModel::Continuous(_) => {
            if dtype.is_continuous() {
                Ok(())
            } else {
                let err = OracleError::InvalidDType {
                    col_ix: col_ix,
                    dtype: dtype.as_string(),
                    expected: String::from("Continuous"),
                };
                Err(err.to_error())
            }
        }
        ColModel::Categorical(_) => {
            if dtype.is_categorical() {
                Ok(())
            } else {
                let err = OracleError::InvalidDType {
                    col_ix: col_ix,
                    dtype: dtype.as_string(),
                    expected: String::from("Categorical"),
                };
                Err(err.to_error())
            }
        }
    }
}

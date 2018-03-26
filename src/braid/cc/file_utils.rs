
use std::fs;
use std::path::Path;
use std::fs::create_dir;
use std::io::{Error, ErrorKind, Result};
use std::collections::BTreeMap;

use cc::{State, FeatureData, Codebook};


/// Returns whether the directory `dir` has a codebook file. Will return
/// `Error` if `dir` does not exist or is not a directory.
pub fn has_codebook(dir: &str) -> Result<bool> {
    let paths = fs::read_dir(dir)?;
    let n_codebooks = paths.fold(0, |acc, path| {
        match path.unwrap().path().extension() {
            Some(s) => if s.to_str().unwrap() == "codebook" {
                acc + 1
            } else { 
                acc 
            },
            None => acc
        }
    });

    match n_codebooks {
        0 => Ok(false),
        1 => Ok(true),
        _ => {
            let err_kind = ErrorKind::InvalidInput;
            Err(Error::new(err_kind, "Too many codebooks"))
        }
    }
}

/// Returns whether the directory `dir` has a data file. Will return
/// `Error` if `dir` does not exist or is not a directory.
pub fn has_data(dir: &str) -> Result<bool> {
    let paths = fs::read_dir(dir)?;
    let n_data_files = paths.fold(0, |acc, path| {
        match path.unwrap().path().extension() {
            Some(s) => if s.to_str().unwrap() == "data" { acc + 1 } else { acc }
            None => acc
        }
    });

    match n_data_files {
        0 => Ok(false),
        1 => Ok(true),
        _ => {
            let err_kind = ErrorKind::InvalidInput;
            Err(Error::new(err_kind, "Too many data files"))
        }
    }
}

/// Returns the list IDs of the states saved in the directory `dir`. Will
/// return an empty vectory if the are no states.  Will return `Error` if `dir` 
/// does not exist or is not a directory.
pub fn get_state_ids(dir: &str) -> Result<Vec<usize>> {
    let paths = fs::read_dir(dir)?;
    let mut state_ids: Vec<usize> = vec![];
    for path in paths {
        let p = path.unwrap().path();
        if let Some(s) = p.extension() {
            if s.to_str().unwrap() == "state" {
                let str_id = p.file_stem().unwrap().to_str().unwrap();
                match str_id.parse::<usize>() {
                    Ok(id) => state_ids.push(id),
                    Err(..) => {
                        let err_kind = ErrorKind::InvalidInput;
                        return Err(Error::new(err_kind, "Invalid state name"))
                    }
                }
            }
        }
    }
    Ok(state_ids)
}

/// Saves all states, the data, and the codebook.
pub fn save_all(
    dir: &str,
    states: BTreeMap<usize, State>,
    data: BTreeMap<usize, FeatureData>,
    codebook: Codebook,
) -> Result<()> {
    unimplemented!();
}

/// Save all the states. Assumes the data and codebook exist.
pub fn save_states(dir: &str, states: BTreeMap<usize, State>) -> Result<()> {
    unimplemented!();
}

/// Saves just some states. Assumes other states, the data and the codebook
/// exist.
pub fn save_state(dir: &str, state: State, id: usize) -> Result<()> {
    unimplemented!();
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR_1: &str = "braid-tests/savedata.valid.1.braid";
    const NO_CODEBOOK_DIR: &str = "braid-tests/savedata.no.codebook.braid";
    const NO_DATA_DIR: &str = "braid-tests/savedata.no.data.braid";

    #[test]
    fn finds_codebook_in_directory_with_codebook() {
        let cb = has_codebook(DIR_1);
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_data_in_directory_with_data() {
        let data = has_data(DIR_1);
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_correct_state_ids() {
        let ids = get_state_ids(DIR_1);
        assert!(ids.is_ok());
        assert_eq!(ids.unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn finds_data_in_no_codebook_dir() {
        let data = has_data(NO_CODEBOOK_DIR);
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_no_codebook_in_no_codebook_dir() {
        let cb = has_codebook(NO_CODEBOOK_DIR);
        assert!(cb.is_ok());
        assert!(!cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_codebook_dir() {
        let ids = get_state_ids(NO_CODEBOOK_DIR);
        assert!(ids.is_ok());
        assert_eq!(ids.unwrap(), vec![0, 1]);
    }

    #[test]
    fn finds_no_data_in_no_data_dir() {
        let data = has_data(NO_DATA_DIR);
        assert!(data.is_ok());
        assert!(!data.unwrap());
    }

    #[test]
    fn finds_codebook_in_no_data_dir() {
        let cb = has_codebook(NO_DATA_DIR);
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_data_dir() {
        let ids = get_state_ids(NO_DATA_DIR);
        assert!(ids.is_ok());
        assert_eq!(ids.unwrap(), vec![0, 1, 2, 3]);
    }
}

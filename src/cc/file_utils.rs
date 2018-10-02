//! Misc file utilities
extern crate env_logger;
extern crate rand;
extern crate serde_yaml;

use self::rand::{FromEntropy, XorShiftRng};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::path::Path;

use cc::{Codebook, FeatureData, State};

/// Returns whether the directory `dir` has a codebook file. Will return
/// `Error` if `dir` does not exist or is not a directory.
pub fn has_codebook(dir: &str) -> Result<bool> {
    let paths = fs::read_dir(dir)?;
    let n_codebooks =
        paths.fold(0, |acc, path| match path.unwrap().path().extension() {
            Some(s) => {
                if s.to_str().unwrap() == "codebook" {
                    acc + 1
                } else {
                    acc
                }
            }
            None => acc,
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
    let n_data_files =
        paths.fold(0, |acc, path| match path.unwrap().path().extension() {
            Some(s) => {
                if s.to_str().unwrap() == "data" {
                    acc + 1
                } else {
                    acc
                }
            }
            None => acc,
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
                        return Err(Error::new(err_kind, "Invalid state name"));
                    }
                }
            }
        }
    }
    Ok(state_ids)
}

pub fn path_validator(dir: &str) -> Result<()> {
    let path = Path::new(dir);
    let err_kind = ErrorKind::InvalidInput;
    if !path.exists() {
        info!("{} does not exist. Creating...", dir);
        fs::create_dir(dir).expect("Could not create directory");
        info!("Done");
        Ok(())
    } else if !path.is_dir() {
        Err(Error::new(err_kind, "Invalid directory"))
    } else {
        Ok(())
    }
}

/// Saves all states, the data, and the codebook.
pub fn save_all(
    dir: &str,
    mut states: &mut BTreeMap<usize, State>,
    data: &BTreeMap<usize, FeatureData>,
    codebook: &Codebook,
) -> Result<()> {
    path_validator(dir)?;
    save_states(dir, &mut states)
        .and_then(|_| save_data(dir, &data))
        .and_then(|_| save_codebook(dir, &codebook))
}

/// Save all the states. Assumes the data and codebook exist.
pub fn save_states(
    dir: &str,
    states: &mut BTreeMap<usize, State>,
) -> Result<()> {
    path_validator(dir)?;
    for (id, state) in states.iter_mut() {
        save_state(dir, state, *id)?;
    }
    Ok(())
}

/// Saves just some states. Assumes other states, the data and the codebook
/// exist.
pub fn save_state(dir: &str, state: &mut State, id: usize) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/{}.state", dir, id);
    let path = Path::new(&filename);
    let data = state.take_data();
    let ser = serde_yaml::to_string(state).unwrap().into_bytes();
    let mut file = fs::File::create(path)?;
    let _nbytes = file.write(&ser)?;
    state.repop_data(data).expect("Could not repopulate data");
    info!("State {} saved to {}", id, filename);
    Ok(())
}

pub fn save_data(dir: &str, data: &BTreeMap<usize, FeatureData>) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/braid.data", dir);
    let path = Path::new(&filename);
    let ser = serde_yaml::to_string(data).unwrap().into_bytes();
    let mut file = fs::File::create(path)?;
    let _nbytes = file.write(&ser)?;
    Ok(())
}

pub fn save_codebook(dir: &str, codebook: &Codebook) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/braid.codebook", dir);
    let path = Path::new(&filename);
    let ser = serde_yaml::to_string(codebook).unwrap().into_bytes();
    let mut file = fs::File::create(path)?;
    let _nbytes = file.write(&ser)?;
    Ok(())
}

pub fn save_rng(dir: &str, rng: &XorShiftRng) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/rng-state.yaml", dir);
    let path = Path::new(&filename);
    let ser = serde_yaml::to_string(rng).unwrap().into_bytes();
    let mut file = fs::File::create(path)?;
    let _nbytes = file.write(&ser)?;
    Ok(())
}

pub fn load_states(dir: &str) -> Result<BTreeMap<usize, State>> {
    let ids = get_state_ids(dir)?;
    let mut states: BTreeMap<usize, State> = BTreeMap::new();
    ids.iter().for_each(|&id| {
        let state = load_state(dir, id).unwrap();
        states.insert(id, state);
    }); // propogate Result
    Ok(states)
}

pub fn load_rng(dir: &str) -> Result<XorShiftRng> {
    let filename = format!("{}/rng-state.yaml", dir);
    let path = Path::new(&filename);
    let rng: XorShiftRng = match fs::File::open(&path) {
        Ok(mut file) => {
            let mut ser = String::new();
            file.read_to_string(&mut ser).unwrap();
            serde_yaml::from_str(&ser.as_str()).unwrap()
        }
        Err(..) => {
            info!("No RNG found, creating default.");
            XorShiftRng::from_entropy()
        }
    };
    Ok(rng)
}

pub fn load_state(dir: &str, id: usize) -> Result<State> {
    let filename = format!("{}/{}.state", dir, id);
    info!("Loading state at {}...", filename);
    let path = Path::new(&filename);
    let mut file = fs::File::open(&path).unwrap();
    let mut ser = String::new();
    file.read_to_string(&mut ser).unwrap();
    let state: State = serde_yaml::from_str(&ser.as_str()).unwrap();
    info!("done");
    Ok(state)
}

pub fn load_codebook(dir: &str) -> Result<Codebook> {
    let filename = format!("{}/braid.codebook", dir);
    info!("Loading codebook at {}...", filename);
    let path = Path::new(&filename);
    let mut file = fs::File::open(&path).unwrap();
    let mut ser = String::new();
    file.read_to_string(&mut ser).unwrap();
    let codebook: Codebook = serde_yaml::from_str(&ser.as_str()).unwrap();
    info!("done");
    Ok(codebook)
}

pub fn load_data(dir: &str) -> Result<BTreeMap<usize, FeatureData>> {
    let filename = format!("{}/braid.data", dir);
    let path = Path::new(&filename);
    let mut file = fs::File::open(&path).unwrap();
    let mut ser = String::new();
    file.read_to_string(&mut ser).unwrap();
    let data = serde_yaml::from_str(&ser.as_str()).unwrap();
    Ok(data)
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

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 3);
        assert!(ids_uw.iter().position(|&x| x == 0).is_some());
        assert!(ids_uw.iter().position(|&x| x == 1).is_some());
        assert!(ids_uw.iter().position(|&x| x == 2).is_some());
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

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 2);
        assert!(ids_uw.iter().position(|&x| x == 0).is_some());
        assert!(ids_uw.iter().position(|&x| x == 1).is_some());
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

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 4);
        assert!(ids_uw.iter().position(|&x| x == 0).is_some());
        assert!(ids_uw.iter().position(|&x| x == 1).is_some());
        assert!(ids_uw.iter().position(|&x| x == 2).is_some());
        assert!(ids_uw.iter().position(|&x| x == 3).is_some());
    }
}

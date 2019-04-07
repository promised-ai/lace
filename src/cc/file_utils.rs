//! Misc file utilities
extern crate bincode;
extern crate braid_codebook;
extern crate log;
extern crate rand;
extern crate serde;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::fs;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::path::Path;

use braid_codebook::codebook::Codebook;
use log::info;
use rand::{FromEntropy, XorShiftRng};
use serde::{Deserialize, Serialize};

use crate::cc::{FeatureData, State};
use crate::interface::file_config::{FileConfig, SerializedType};

fn save_as_type<T: Serialize>(
    obj: &T,
    filename: &String,
    serialized_type: SerializedType,
) -> Result<()> {
    let path = Path::new(&filename);
    let bytes: Vec<u8> = match serialized_type {
        SerializedType::Yaml => {
            serde_yaml::to_string(&obj).unwrap().into_bytes()
        }
        SerializedType::Bincode => bincode::serialize(&obj).unwrap(),
    };
    let mut file = fs::File::create(path)?;
    let _nbytes = file.write(&bytes)?;
    Ok(())
}

fn load_as_type<T>(
    filename: &String,
    serialized_type: SerializedType,
) -> Result<T>
where
    for<'de> T: Deserialize<'de>,
{
    let path = Path::new(&filename);
    let mut file = fs::File::open(&path)?;

    let obj: T = match serialized_type {
        SerializedType::Yaml => {
            let mut ser = String::new();
            file.read_to_string(&mut ser).unwrap();
            serde_yaml::from_str(&ser.as_str()).unwrap()
        }
        SerializedType::Bincode => bincode::deserialize_from(&file).unwrap(),
    };

    Ok(obj)
}

/// Load the file config
pub fn load_file_config(filename: &String) -> Result<FileConfig> {
    load_as_type(&filename, SerializedType::Yaml)
}

/// Load the file config
pub fn save_file_config(
    file_config: &FileConfig,
    filename: &String,
) -> Result<()> {
    save_as_type(&file_config, &filename, SerializedType::Yaml)
}

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
    file_config: &FileConfig,
) -> Result<()> {
    path_validator(dir)?;
    save_states(dir, &mut states, &file_config)
        .and_then(|_| save_data(dir, &data, &file_config))
        .and_then(|_| save_codebook(dir, &codebook))
}

/// Save all the states. Assumes the data and codebook exist.
pub fn save_states(
    dir: &str,
    states: &mut BTreeMap<usize, State>,
    file_config: &FileConfig,
) -> Result<()> {
    path_validator(dir)?;
    for (id, state) in states.iter_mut() {
        save_state(dir, state, *id, &file_config)?;
    }
    Ok(())
}

/// Saves just some states. Assumes other states, the data and the codebook
/// exist.
pub fn save_state(
    dir: &str,
    state: &mut State,
    id: usize,
    file_config: &FileConfig,
) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/{}.state", dir, id);
    let serialized_type = file_config.serialized_type.unwrap_or_default();

    let data = state.take_data();
    save_as_type(&state, &filename, serialized_type)?;
    state.repop_data(data).expect("Could not repopulate data");

    info!("State {} saved to {}", id, filename);
    Ok(())
}

pub fn save_data(
    dir: &str,
    data: &BTreeMap<usize, FeatureData>,
    file_config: &FileConfig,
) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/braid.data", dir);
    let serialized_type = file_config.serialized_type.unwrap_or_default();
    save_as_type(&data, &filename, serialized_type)
}

pub fn save_codebook(dir: &str, codebook: &Codebook) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/braid.codebook", dir);
    save_as_type(&codebook, &filename, SerializedType::default())
}

pub fn save_rng(dir: &str, rng: &XorShiftRng) -> Result<()> {
    path_validator(dir)?;
    let filename = format!("{}/rng-state.yaml", dir);
    save_as_type(&rng, &filename, SerializedType::default())
}

pub fn load_states(
    dir: &str,
    file_config: &FileConfig,
) -> Result<BTreeMap<usize, State>> {
    let ids = get_state_ids(dir)?;
    let mut states: BTreeMap<usize, State> = BTreeMap::new();
    ids.iter().for_each(|&id| {
        let state = load_state(dir, id, &file_config).unwrap();
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

pub fn load_state(
    dir: &str,
    id: usize,
    file_config: &FileConfig,
) -> Result<State> {
    let filename = format!("{}/{}.state", dir, id);
    info!("Loading state at {}...", filename);
    let serialized_type = file_config.serialized_type.unwrap_or_default();
    load_as_type(&filename, serialized_type)
}

pub fn load_codebook(dir: &str) -> Result<Codebook> {
    let filename = format!("{}/braid.codebook", dir);
    load_as_type(&filename, SerializedType::Yaml)
}

pub fn load_data(
    dir: &str,
    file_config: &FileConfig,
) -> Result<BTreeMap<usize, FeatureData>> {
    let filename = format!("{}/braid.data", dir);
    let serialized_type = file_config.serialized_type.unwrap_or_default();
    let data: BTreeMap<usize, FeatureData> =
        load_as_type(&filename, serialized_type)?;
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
        println!("{:?}", ids);
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 4);
        assert!(ids_uw.iter().position(|&x| x == 0).is_some());
        assert!(ids_uw.iter().position(|&x| x == 1).is_some());
        assert!(ids_uw.iter().position(|&x| x == 2).is_some());
        assert!(ids_uw.iter().position(|&x| x == 3).is_some());
    }
}

//! Misc file utilities
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::path::{Path, PathBuf};

use braid_codebook::Codebook;
use log::info;
use serde::{Deserialize, Serialize};

use crate::cc::{FeatureData, State};
use crate::file_config::{FileConfig, SerializedType};

fn save_as_type<T: Serialize>(
    obj: &T,
    path: &Path,
    serialized_type: SerializedType,
) -> io::Result<()> {
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

fn load_as_type<T>(path: &Path, serialized_type: SerializedType) -> Result<T>
where
    for<'de> T: Deserialize<'de>,
{
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
pub fn load_file_config(dir: &Path) -> Result<FileConfig> {
    let config_path = get_config_path(dir);
    load_as_type(config_path.as_path(), SerializedType::Yaml)
}

/// Load the file config
pub fn save_file_config(dir: &Path, file_config: &FileConfig) -> Result<()> {
    let config_path = get_config_path(dir);
    save_as_type(&file_config, config_path.as_path(), SerializedType::Yaml)
}

/// Count the number of files in a directory with a given extension, `ext`
fn ext_count(dir: &Path, ext: &str) -> Result<u32> {
    let paths = fs::read_dir(dir)?;
    let n =
        paths.fold(0_u32, |acc, path| match path.unwrap().path().extension() {
            Some(s) => {
                if s.to_str().unwrap() == ext {
                    acc + 1
                } else {
                    acc
                }
            }
            None => acc,
        });
    Ok(n)
}

/// Returns whether the directory `dir` has a codebook file. Will return
/// `Error` if `dir` does not exist or is not a directory.
pub fn has_codebook(dir: &Path) -> Result<bool> {
    let n_codebooks = ext_count(dir, "codebook")?;
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
pub fn has_data(dir: &Path) -> Result<bool> {
    let n_data_files = ext_count(dir, "data")?;
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
pub fn get_state_ids(dir: &Path) -> Result<Vec<usize>> {
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

pub fn path_validator(path: &Path) -> io::Result<()> {
    if !path.exists() {
        info!("{} does not exist. Creating...", path.to_str().unwrap());
        fs::create_dir(path).expect("Could not create directory");
        info!("Done");
        Ok(())
    } else if !path.is_dir() {
        let kind = io::ErrorKind::InvalidInput;
        Err(io::Error::new(kind, "path is not a directory"))
    } else {
        Ok(())
    }
}

/// Saves all states, the data, and the codebook.
pub fn save_all(
    dir: &Path,
    mut states: &mut Vec<State>,
    state_ids: &Vec<usize>,
    data: &BTreeMap<usize, FeatureData>,
    codebook: &Codebook,
    file_config: &FileConfig,
) -> io::Result<()> {
    path_validator(dir)?;
    save_states(dir, &mut states, &state_ids, &file_config)
        .and_then(|_| save_data(dir, &data, &file_config))
        .and_then(|_| save_codebook(dir, &codebook))
}

/// Save all the states. Assumes the data and codebook exist.
pub fn save_states(
    dir: &Path,
    states: &mut Vec<State>,
    state_ids: &Vec<usize>,
    file_config: &FileConfig,
) -> io::Result<()> {
    path_validator(dir)?;
    for (state, id) in states.iter_mut().zip(state_ids.iter()) {
        save_state(dir, state, *id, &file_config)?;
    }
    Ok(())
}

fn get_state_path(dir: &Path, state_id: usize) -> PathBuf {
    let mut state_path = PathBuf::from(dir);
    state_path.push(format!("{}", state_id));
    state_path.set_extension("state");

    state_path
}

fn get_data_path(dir: &Path) -> PathBuf {
    let mut data_path = PathBuf::from(dir);
    data_path.push("braid");
    data_path.set_extension("data");

    data_path
}

fn get_codebook_path(dir: &Path) -> PathBuf {
    let mut cb_path = PathBuf::from(dir);
    cb_path.push("braid");
    cb_path.set_extension("codebook");

    cb_path
}

pub fn get_config_path(dir: &Path) -> PathBuf {
    let mut config_path = PathBuf::from(dir);
    config_path.push("config");
    config_path.set_extension("yaml");

    config_path
}

/// Saves just some states. Assumes other states, the data and the codebook
/// exist.
pub fn save_state(
    dir: &Path,
    state: &mut State,
    state_id: usize,
    file_config: &FileConfig,
) -> io::Result<()> {
    path_validator(dir)?;
    let state_path = get_state_path(dir, state_id);

    let serialized_type = file_config.serialized_type.unwrap_or_default();

    let data = state.take_data();
    save_as_type(&state, state_path.as_path(), serialized_type)?;
    state.repop_data(data);

    info!("State {} saved to {:?}", state_id, state_path);
    Ok(())
}

pub fn save_data(
    dir: &Path,
    data: &BTreeMap<usize, FeatureData>,
    file_config: &FileConfig,
) -> Result<()> {
    path_validator(dir)?;
    let data_path = get_data_path(dir);
    let serialized_type = file_config.serialized_type.unwrap_or_default();
    save_as_type(&data, data_path.as_path(), serialized_type)
}

pub fn save_codebook(dir: &Path, codebook: &Codebook) -> Result<()> {
    path_validator(dir)?;
    let cb_path = get_codebook_path(dir);
    save_as_type(&codebook, cb_path.as_path(), SerializedType::default())
}

/// Return (states, state_ids) tuple
pub fn load_states(
    dir: &Path,
    file_config: &FileConfig,
) -> Result<(Vec<State>, Vec<usize>)> {
    let state_ids = get_state_ids(dir)?;
    let states: Result<Vec<_>> = state_ids
        .iter()
        .map(|&id| load_state(dir, id, &file_config))
        .collect();

    states.and_then(|s| Ok((s, state_ids)))
}

pub fn load_state(
    dir: &Path,
    state_id: usize,
    file_config: &FileConfig,
) -> Result<State> {
    let state_path = get_state_path(dir, state_id);
    info!("Loading state at {:?}...", state_path);
    let serialized_type = file_config.serialized_type.unwrap_or_default();
    load_as_type(state_path.as_path(), serialized_type)
}

pub fn load_codebook(dir: &Path) -> Result<Codebook> {
    let codebook_path = get_codebook_path(dir);
    load_as_type(codebook_path.as_path(), SerializedType::Yaml)
}

pub fn load_data(
    dir: &Path,
    file_config: &FileConfig,
) -> Result<BTreeMap<usize, FeatureData>> {
    let data_path = get_data_path(dir);
    let serialized_type = file_config.serialized_type.unwrap_or_default();
    let data: BTreeMap<usize, FeatureData> =
        load_as_type(data_path.as_path(), serialized_type)?;
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
        let cb = has_codebook(Path::new(DIR_1));
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_data_in_directory_with_data() {
        let data = has_data(Path::new(DIR_1));
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_correct_state_ids() {
        let ids = get_state_ids(Path::new(DIR_1));
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 3);
        assert!(ids_uw.iter().position(|&x| x == 0).is_some());
        assert!(ids_uw.iter().position(|&x| x == 1).is_some());
        assert!(ids_uw.iter().position(|&x| x == 2).is_some());
    }

    #[test]
    fn finds_data_in_no_codebook_dir() {
        let data = has_data(Path::new(NO_CODEBOOK_DIR));
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_no_codebook_in_no_codebook_dir() {
        let cb = has_codebook(Path::new(NO_CODEBOOK_DIR));
        assert!(cb.is_ok());
        assert!(!cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_codebook_dir() {
        let ids = get_state_ids(Path::new(NO_CODEBOOK_DIR));
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 2);
        assert!(ids_uw.iter().position(|&x| x == 0).is_some());
        assert!(ids_uw.iter().position(|&x| x == 1).is_some());
    }

    #[test]
    fn finds_no_data_in_no_data_dir() {
        let data = has_data(Path::new(NO_DATA_DIR));
        assert!(data.is_ok());
        assert!(!data.unwrap());
    }

    #[test]
    fn finds_codebook_in_no_data_dir() {
        let cb = has_codebook(Path::new(NO_DATA_DIR));
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_data_dir() {
        let ids = get_state_ids(Path::new(NO_DATA_DIR));
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

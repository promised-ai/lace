//! Misc file utilities
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::path::{Path, PathBuf};

use braid_cc::state::State;
use braid_codebook::Codebook;
use braid_data::FeatureData;
use log::info;
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};

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
    let mut file = io::BufWriter::new(fs::File::create(path)?);
    let _nbytes = file.write(&bytes)?;
    Ok(())
}

fn load_as_type<T>(path: &Path, serialized_type: SerializedType) -> Result<T>
where
    for<'de> T: Deserialize<'de>,
{
    let mut file = io::BufReader::new(fs::File::open(&path)?);

    let obj: T = match serialized_type {
        SerializedType::Yaml => {
            let mut ser = String::new();
            file.read_to_string(&mut ser).unwrap();
            serde_yaml::from_str(ser.as_str()).unwrap()
        }
        SerializedType::Bincode => bincode::deserialize_from(file).unwrap(),
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn has_codebook(dir: &Path) -> Result<bool> {
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
#[allow(dead_code)]
pub(crate) fn has_data(dir: &Path) -> Result<bool> {
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
        let p = path?;
        // do not try to load directories
        if p.file_type()?.is_file() {
            let pathbuf = p.path();
            let ext = match pathbuf.extension() {
                Some(ext) => ext.to_str().unwrap(),
                None => continue,
            };

            // state files end in .state
            if ext == "state" {
                if let Some(stem) = pathbuf.file_stem() {
                    let str_id = stem.to_str().unwrap();

                    // state file names should parse to usize
                    match str_id.parse::<usize>() {
                        Ok(id) => state_ids.push(id),
                        Err(..) => {
                            let err_kind = ErrorKind::InvalidInput;
                            let msg = format!(
                                "Invalid file name for state '{}'. \
                                States names must parse to usize.",
                                str_id
                            );
                            return Err(Error::new(err_kind, msg));
                        }
                    }
                } else {
                    continue;
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

/// Save all the states. Assumes the data and codebook exist.
pub fn save_states(
    dir: &Path,
    states: &mut Vec<State>,
    state_ids: &[usize],
    file_config: &FileConfig,
) -> io::Result<()> {
    path_validator(dir)?;
    for (state, id) in states.iter_mut().zip(state_ids.iter()) {
        save_state(dir, state, *id, file_config)?;
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

fn get_rng_path(dir: &Path) -> PathBuf {
    let mut rng_path = PathBuf::from(dir);
    rng_path.push("braid");
    rng_path.set_extension("rng");

    rng_path
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

pub fn save_rng(dir: &Path, rng: &Xoshiro256Plus) -> Result<()> {
    path_validator(dir)?;
    let rng_path = get_rng_path(dir);
    save_as_type(&rng, rng_path.as_path(), SerializedType::default())
}

/// Return (states, state_ids) tuple
pub fn load_states(
    dir: &Path,
    file_config: &FileConfig,
) -> Result<(Vec<State>, Vec<usize>)> {
    let state_ids = get_state_ids(dir)?;
    let states: Result<Vec<_>> = state_ids
        .iter()
        .map(|&id| load_state(dir, id, file_config))
        .collect();

    states.map(|s| (s, state_ids))
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

pub fn load_rng(dir: &Path) -> Result<Xoshiro256Plus> {
    let rng_path = get_rng_path(dir);
    load_as_type(rng_path.as_path(), SerializedType::Yaml)
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
    use tempfile::TempDir;

    const VALID_FILES: [&str; 5] = [
        "0.state",
        "1.state",
        "2.state",
        "test.codebook",
        "test.data",
    ];

    // puppy.state is not a valid state name
    const BAD_STATE_FILES: [&str; 5] = [
        "puppy.state",
        "1.state",
        "2.state",
        "test.codebook",
        "test.data",
    ];

    // crates a dir that has a .state extension. Not valid.
    const STATE_DIR_FILES: [&str; 5] = [
        "0.state/empty.txt",
        "1.state",
        "2.state",
        "test.codebook",
        "test.data",
    ];

    const NO_DATA_FILES: [&str; 5] =
        ["0.state", "1.state", "2.state", "3.state", "test.codebook"];

    const NO_CODEBOOK_FILES: [&str; 3] = ["0.state", "1.state", "test.data"];

    fn create_braidfile(fnames: &[&str]) -> TempDir {
        let dir = TempDir::new().unwrap();
        fnames.iter().for_each(|fname| {
            let _f = fs::File::create(dir.path().join(fname));
        });
        dir
    }

    #[test]
    fn finds_codebook_in_directory_with_codebook() {
        let dir = create_braidfile(&VALID_FILES);
        let cb = has_codebook(dir.path());
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_data_in_directory_with_data() {
        let dir = create_braidfile(&VALID_FILES);
        let data = has_data(dir.path());
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_correct_state_ids() {
        let dir = create_braidfile(&VALID_FILES);
        let ids = get_state_ids(dir.path());
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 3);
        assert!(ids_uw.iter().any(|&x| x == 0));
        assert!(ids_uw.iter().any(|&x| x == 1));
        assert!(ids_uw.iter().any(|&x| x == 2));
    }

    #[test]
    fn bad_state_file_errs() {
        let dir = create_braidfile(&BAD_STATE_FILES);
        let err = get_state_ids(dir.path()).unwrap_err();
        assert!(err.to_string().contains("puppy"));
    }

    #[test]
    fn finds_correct_state_ids_with_dir_with_state_extension() {
        let dir = create_braidfile(&STATE_DIR_FILES);
        let ids = get_state_ids(dir.path());
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 2);
        assert!(ids_uw.iter().any(|&x| x == 1));
        assert!(ids_uw.iter().any(|&x| x == 2));
    }

    #[test]
    fn finds_data_in_no_codebook_dir() {
        let dir = create_braidfile(&NO_CODEBOOK_FILES);
        let data = has_data(dir.path());
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_no_codebook_in_no_codebook_dir() {
        let dir = create_braidfile(&NO_CODEBOOK_FILES);
        let cb = has_codebook(dir.path());
        assert!(cb.is_ok());
        assert!(!cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_codebook_dir() {
        let dir = create_braidfile(&NO_CODEBOOK_FILES);
        let ids = get_state_ids(dir.path());
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 2);
        assert!(ids_uw.iter().any(|&x| x == 0));
        assert!(ids_uw.iter().any(|&x| x == 1));
    }

    #[test]
    fn finds_no_data_in_no_data_dir() {
        let dir = create_braidfile(&NO_DATA_FILES);
        let data = has_data(dir.path());
        assert!(data.is_ok());
        assert!(!data.unwrap());
    }

    #[test]
    fn finds_codebook_in_no_data_dir() {
        let dir = create_braidfile(&NO_DATA_FILES);
        let cb = has_codebook(dir.path());
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_data_dir() {
        let dir = create_braidfile(&NO_DATA_FILES);
        let ids = get_state_ids(dir.path());
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 4);
        assert!(ids_uw.iter().any(|&x| x == 0));
        assert!(ids_uw.iter().any(|&x| x == 1));
        assert!(ids_uw.iter().any(|&x| x == 2));
        assert!(ids_uw.iter().any(|&x| x == 3));
    }
}

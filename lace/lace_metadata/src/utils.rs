//! Misc file utilities
use std::fs;
use std::io;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use log::info;
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};

use crate::latest::Codebook;
use crate::versions::v1::DataStore;
use crate::versions::v2::DatalessState;
use crate::{Error, FileConfig, SerializedType};

fn extenson_from_path<P: AsRef<Path>>(path: &P) -> Result<&str, Error> {
    path.as_ref()
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid file type",
            ))
        })
}

fn serialized_type_from_path<P: AsRef<Path>>(
    path: &P,
) -> Result<SerializedType, Error> {
    let ext = extenson_from_path(path)?;
    SerializedType::from_str(ext)
}

pub fn serialize_obj<T, P>(obj: &T, path: P) -> Result<(), Error>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let serialized_type = serialized_type_from_path(&path)?;

    save(obj, path, serialized_type)
}

pub fn deserialize_file<T, P>(path: P) -> Result<T, Error>
where
    for<'de> T: Deserialize<'de>,
    P: AsRef<Path>,
{
    let serialized_type = serialized_type_from_path(&path)?;

    load(path, serialized_type)
}

pub fn save<T, P>(
    obj: &T,
    path: P,
    serialized_type: SerializedType,
) -> Result<(), Error>
where
    T: Serialize,
    P: AsRef<Path>,
{
    match serialized_type {
        SerializedType::Yaml => serde_yaml::to_string(&obj)
            .map_err(Error::Yaml)
            .map(|s| s.into_bytes()),
        SerializedType::Json => {
            serde_json::to_vec_pretty(&obj).map_err(Error::Json)
        }
        SerializedType::Bincode => {
            bincode::serialize(&obj).map_err(Error::Bincode)
        }
    }
    .and_then(|bytes| {
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut writer = io::BufWriter::new(file);
        writer.write_all(&bytes).map_err(Error::Io)
    })
}

fn save_as_type<T: Serialize, P: AsRef<Path>>(
    obj: &T,
    path: P,
    serialized_type: SerializedType,
) -> Result<(), Error> {
    match serialized_type {
        SerializedType::Yaml => serde_yaml::to_string(&obj)
            .map_err(Error::Yaml)
            .map(|s| s.into_bytes()),
        SerializedType::Json => {
            serde_json::to_vec_pretty(&obj).map_err(Error::Json)
        }
        SerializedType::Bincode => {
            bincode::serialize(&obj).map_err(Error::Bincode)
        }
    }
    .and_then(|bytes| {
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut writer = io::BufWriter::new(file);
        writer.write_all(&bytes).map_err(Error::Io)
    })
}

pub(crate) fn load<T, P>(
    path: P,
    serialized_type: SerializedType,
) -> Result<T, Error>
where
    for<'de> T: Deserialize<'de>,
    P: AsRef<Path>,
{
    let mut file = io::BufReader::new(fs::File::open(path)?);

    match serialized_type {
        SerializedType::Yaml => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_yaml::from_str(ser.as_str()).map_err(Error::Yaml)
        }
        SerializedType::Json => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_json::from_str(ser.as_str()).map_err(Error::Json)
        }
        SerializedType::Bincode => {
            bincode::deserialize_from(file).map_err(Error::Bincode)
        }
    }
}

pub(crate) fn load_as_type<T, P>(
    path: P,
    serialized_type: SerializedType,
) -> Result<T, Error>
where
    for<'de> T: Deserialize<'de>,
    P: AsRef<Path>,
{
    let mut file = io::BufReader::new(fs::File::open(path.as_ref())?);

    match serialized_type {
        SerializedType::Yaml => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_yaml::from_str(ser.as_str()).map_err(Error::Yaml)
        }
        SerializedType::Bincode => {
            bincode::deserialize_from(file).map_err(Error::Bincode)
        }
        SerializedType::Json => {
            serde_json::from_reader(file).map_err(Error::Json)
        }
    }
}

pub fn path_validator<P: AsRef<Path>>(path: P) -> Result<(), Error> {
    if !path.as_ref().exists() {
        info!(
            "{} does not exist. Creating...",
            path.as_ref().to_str().unwrap()
        );
        fs::create_dir(path).map_err(Error::Io)
    } else if !path.as_ref().is_dir() {
        let kind = io::ErrorKind::InvalidInput;
        Err(io::Error::new(kind, "path is not a directory").into())
    } else {
        Ok(())
    }
}

pub(crate) fn get_state_path<P: AsRef<Path>>(
    path: P,
    state_id: usize,
) -> PathBuf {
    let mut state_path = PathBuf::from(path.as_ref());
    state_path.push(state_id.to_string());
    state_path.set_extension("state");

    state_path
}

pub(crate) fn get_data_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut data_path = PathBuf::from(path.as_ref());
    data_path.push("lace");
    data_path.set_extension("data");

    data_path
}

pub(crate) fn get_codebook_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut cb_path = PathBuf::from(path.as_ref());
    cb_path.push("lace");
    cb_path.set_extension("codebook");

    cb_path
}

pub(crate) fn get_rng_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut rng_path = PathBuf::from(path.as_ref());
    rng_path.push("rng");
    rng_path.set_extension("yaml");

    rng_path
}

pub(crate) fn get_config_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut config_path = PathBuf::from(path.as_ref());
    config_path.push("config");
    config_path.set_extension("yaml");

    config_path
}

/// Returns the list IDs of the states saved in the directory `dir`. Will
/// return an empty vectory if the are no states.  Will return `Error` if `dir`
/// does not exist or is not a directory.
pub fn get_state_ids<P: AsRef<Path>>(path: P) -> Result<Vec<usize>, Error> {
    let paths = fs::read_dir(path)?;
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
                            let path_str = pathbuf
                                .into_os_string()
                                .into_string()
                                .unwrap_or_else(|_| {
                                    String::from("<InvalidString>")
                                });
                            return Err(Error::StateFileNameInvalid(path_str));
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

pub fn save_state<P: AsRef<Path>>(
    path: P,
    state: &DatalessState,
    state_id: usize,
    file_config: &FileConfig,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let state_path = get_state_path(path, state_id);

    let serialized_type = file_config.serialized_type;

    save(state, state_path.as_path(), serialized_type)?;

    info!("State {} saved to {:?}", state_id, state_path);
    Ok(())
}

/// Save all the states. Assumes the data and codebook exist.
pub(crate) fn save_states<P: AsRef<Path>>(
    path: P,
    states: &[DatalessState],
    state_ids: &[usize],
    file_config: &FileConfig,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    states
        .iter()
        .zip(state_ids.iter())
        .try_for_each(|(state, id)| {
            save_state(path.as_ref(), state, *id, file_config)
        })
}

pub(crate) fn save_data<P: AsRef<Path>>(
    path: P,
    data: &DataStore,
    file_config: &FileConfig,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let data_path = get_data_path(path);
    save(data, data_path, file_config.serialized_type)
}

pub(crate) fn save_codebook<P: AsRef<Path>>(
    path: P,
    codebook: &Codebook,
    file_config: &FileConfig,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let cb_path = get_codebook_path(path);
    save(codebook, cb_path, file_config.serialized_type)
}

pub(crate) fn save_rng<P: AsRef<Path>>(
    path: P,
    rng: &Xoshiro256Plus,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let rng_path = get_rng_path(path);
    save_as_type(&rng, rng_path, SerializedType::Yaml)
}

/// Load the file config
pub fn load_file_config<P: AsRef<Path>>(path: P) -> Result<FileConfig, Error> {
    let config_path = get_config_path(path);
    load_as_type(config_path, SerializedType::Yaml)
}

/// Load the file config
pub fn save_file_config<P: AsRef<Path>>(
    path: P,
    file_config: &FileConfig,
) -> Result<(), Error> {
    let config_path = get_config_path(path);
    save_as_type(&file_config, config_path, SerializedType::Yaml)
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

    /// Count the number of files in a directory with a given extension, `ext`
    fn ext_count(dir: &Path, ext: &str) -> io::Result<u32> {
        let paths = fs::read_dir(dir)?;
        let n = paths.fold(0_u32, |acc, path| {
            match path.unwrap().path().extension() {
                Some(s) => {
                    if s.to_str().unwrap() == ext {
                        acc + 1
                    } else {
                        acc
                    }
                }
                None => acc,
            }
        });
        Ok(n)
    }

    /// Returns whether the directory `dir` has a codebook file. Will return
    /// `Error` if `dir` does not exist or is not a directory.
    fn has_codebook(dir: &Path) -> io::Result<bool> {
        let n_codebooks = ext_count(dir, "codebook")?;
        match n_codebooks {
            0 => Ok(false),
            1 => Ok(true),
            _ => {
                let err_kind = io::ErrorKind::InvalidInput;
                Err(io::Error::new(err_kind, "Too many codebooks"))
            }
        }
    }

    /// Returns whether the directory `dir` has a data file. Will return
    /// `Error` if `dir` does not exist or is not a directory.
    fn has_data(dir: &Path) -> io::Result<bool> {
        let n_data_files = ext_count(dir, "data")?;
        match n_data_files {
            0 => Ok(false),
            1 => Ok(true),
            _ => {
                let err_kind = io::ErrorKind::InvalidInput;
                Err(io::Error::new(err_kind, "Too many data files"))
            }
        }
    }

    fn create_lacefile(fnames: &[&str]) -> TempDir {
        let dir = TempDir::new().unwrap();
        fnames.iter().for_each(|fname| {
            let _f = fs::File::create(dir.path().join(fname));
        });
        dir
    }

    #[test]
    fn finds_codebook_in_directory_with_codebook() {
        let dir = create_lacefile(&VALID_FILES);
        let cb = has_codebook(dir.path());
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_data_in_directory_with_data() {
        let dir = create_lacefile(&VALID_FILES);
        let data = has_data(dir.path());
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_correct_state_ids() {
        let dir = create_lacefile(&VALID_FILES);
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
        let dir = create_lacefile(&BAD_STATE_FILES);
        let err = get_state_ids(dir.path()).unwrap_err();
        assert!(err.to_string().contains("puppy"));
    }

    #[test]
    fn finds_correct_state_ids_with_dir_with_state_extension() {
        let dir = create_lacefile(&STATE_DIR_FILES);
        let ids = get_state_ids(dir.path());
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 2);
        assert!(ids_uw.iter().any(|&x| x == 1));
        assert!(ids_uw.iter().any(|&x| x == 2));
    }

    #[test]
    fn finds_data_in_no_codebook_dir() {
        let dir = create_lacefile(&NO_CODEBOOK_FILES);
        let data = has_data(dir.path());
        assert!(data.is_ok());
        assert!(data.unwrap());
    }

    #[test]
    fn finds_no_codebook_in_no_codebook_dir() {
        let dir = create_lacefile(&NO_CODEBOOK_FILES);
        let cb = has_codebook(dir.path());
        assert!(cb.is_ok());
        assert!(!cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_codebook_dir() {
        let dir = create_lacefile(&NO_CODEBOOK_FILES);
        let ids = get_state_ids(dir.path());
        assert!(ids.is_ok());

        let ids_uw = ids.unwrap();
        assert_eq!(ids_uw.len(), 2);
        assert!(ids_uw.iter().any(|&x| x == 0));
        assert!(ids_uw.iter().any(|&x| x == 1));
    }

    #[test]
    fn finds_no_data_in_no_data_dir() {
        let dir = create_lacefile(&NO_DATA_FILES);
        let data = has_data(dir.path());
        assert!(data.is_ok());
        assert!(!data.unwrap());
    }

    #[test]
    fn finds_codebook_in_no_data_dir() {
        let dir = create_lacefile(&NO_DATA_FILES);
        let cb = has_codebook(dir.path());
        assert!(cb.is_ok());
        assert!(cb.unwrap());
    }

    #[test]
    fn finds_correct_ids_in_no_data_dir() {
        let dir = create_lacefile(&NO_DATA_FILES);
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

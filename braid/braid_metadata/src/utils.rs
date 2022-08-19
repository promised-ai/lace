//! Misc file utilities
use std::convert::TryFrom;
use std::fmt::Display;
use std::fs;
use std::io;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use log::info;
use rand_xoshiro::Xoshiro256Plus;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, CHACHA20_POLY1305};
use serde::{Deserialize, Serialize};

use crate::error::TomlError;
use crate::latest::{Codebook, DatalessState};
use crate::versions::v1::DataStore;
use crate::{Error, FileConfig, SerializedType};

fn generate_nonce() -> Result<[u8; 12], Error> {
    use ring::rand::SecureRandom;

    let rng = ring::rand::SystemRandom::new();
    let mut value: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    rng.fill(&mut value)?;
    Ok(value)
}

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

    save_as_possibly_encrypted(obj, path, serialized_type, None)
}

pub fn deserialize_file<T, P>(path: P) -> Result<T, Error>
where
    for<'de> T: Deserialize<'de>,
    P: AsRef<Path>,
{
    let serialized_type = serialized_type_from_path(&path)?;

    load_as_possibly_encrypted(path, serialized_type, None)
}

/// An ecryption and decryption key for Braid metadata and data.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub struct EncryptionKey([u8; 32]);

impl FromStr for EncryptionKey {
    type Err = hex::FromHexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut raw: [u8; 32] = [0; 32];
        hex::decode_to_slice(s, &mut raw)?;
        Ok(Self(raw))
    }
}

impl From<[u8; 32]> for EncryptionKey {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl From<EncryptionKey> for String {
    fn from(key: EncryptionKey) -> Self {
        key.to_string()
    }
}

impl TryFrom<String> for EncryptionKey {
    type Error = <EncryptionKey as FromStr>::Err;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::from_str(&s)
    }
}

impl AsRef<[u8; 32]> for EncryptionKey {
    fn as_ref(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Display for EncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(&self.0).fmt(f)
    }
}

fn serialize_and_encrypt<T>(
    key: &EncryptionKey,
    obj: &T,
) -> Result<Vec<u8>, Error>
where
    T: Serialize,
{
    // Serialize the object into bytes
    let mut bytes = bincode::serialize(&obj).map_err(Error::Bincode)?;

    // Create a key
    let ub_key = UnboundKey::new(&CHACHA20_POLY1305, key.as_ref())?;
    let key = LessSafeKey::new(ub_key);

    // generate the nonce and encrypt the data
    let nonce = generate_nonce()?;
    key.seal_in_place_append_tag(
        Nonce::try_assume_unique_for_key(&nonce)?,
        Aad::empty(),
        &mut bytes,
    )?;

    // Append the nonce to the back of the encrypted data
    nonce.iter().for_each(|&b| bytes.push(b));

    Ok(bytes)
}

pub fn save_as_possibly_encrypted<T, P>(
    obj: &T,
    path: P,
    serialized_type: SerializedType,
    key: Option<&EncryptionKey>,
) -> Result<(), Error>
where
    T: Serialize,
    P: AsRef<Path>,
{
    match (serialized_type, key) {
        (SerializedType::Yaml, _) => serde_yaml::to_string(&obj)
            .map_err(Error::Yaml)
            .map(|s| s.into_bytes()),
        (SerializedType::Toml, _) => {
            toml::to_vec(&obj).map_err(|err| Error::Toml(TomlError::from(err)))
        }
        (SerializedType::Json, _) => {
            serde_json::to_vec_pretty(&obj).map_err(Error::Json)
        }
        (SerializedType::Bincode, _) => {
            bincode::serialize(&obj).map_err(Error::Bincode)
        }
        (SerializedType::Encrypted, Some(key)) => {
            serialize_and_encrypt(key, obj)
        }
        (SerializedType::Encrypted, None) => Err(Error::EncryptionKeyNotFound),
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
        SerializedType::Toml => {
            toml::to_vec(&obj).map_err(|err| Error::Toml(TomlError::from(err)))
        }
        SerializedType::Json => {
            serde_json::to_vec_pretty(&obj).map_err(Error::Json)
        }
        SerializedType::Bincode => {
            bincode::serialize(&obj).map_err(Error::Bincode)
        }
        SerializedType::Encrypted => {
            panic!(
                "We messed up. Somehow we tried to encrypt a file ({:?}) that \
                does not support encryption",
                path.as_ref()
            )
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

fn load_encrypted<T>(
    key: &EncryptionKey,
    mut bytes: Vec<u8>,
) -> Result<T, Error>
where
    for<'de> T: Deserialize<'de>,
{
    // Create the key
    let ub_key = UnboundKey::new(&CHACHA20_POLY1305, key.as_ref())?;
    let opening_key = LessSafeKey::new(ub_key);

    // Get the nonce, which is the last 12 bytes of the file
    let n = bytes.len();
    let value = bytes.split_off(n - 12);
    let nonce = Nonce::try_assume_unique_for_key(value.as_slice())?;

    // Try to open and deserialize
    opening_key.open_in_place(nonce, Aad::empty(), &mut bytes)?;
    let obj = bincode::deserialize(&bytes)?;
    Ok(obj)
}

pub(crate) fn load_as_possibly_encrypted<T, P>(
    path: P,
    serialized_type: SerializedType,
    key: Option<&EncryptionKey>,
) -> Result<T, Error>
where
    for<'de> T: Deserialize<'de>,
    P: AsRef<Path>,
{
    let mut file = io::BufReader::new(fs::File::open(path)?);

    match (serialized_type, key) {
        (SerializedType::Yaml, _) => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_yaml::from_str(ser.as_str()).map_err(Error::Yaml)
        }
        (SerializedType::Toml, _) => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            toml::from_str(ser.as_str())
                .map_err(|err| Error::Toml(TomlError::from(err)))
        }
        (SerializedType::Json, _) => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            serde_json::from_str(ser.as_str()).map_err(Error::Json)
        }
        (SerializedType::Bincode, _) => {
            bincode::deserialize_from(file).map_err(Error::Bincode)
        }
        (SerializedType::Encrypted, Some(key)) => {
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            load_encrypted(key, bytes)
        }
        (SerializedType::Encrypted, None) => Err(Error::EncryptionKeyNotFound),
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
        SerializedType::Toml => {
            let mut ser = String::new();
            file.read_to_string(&mut ser)?;
            toml::from_str(ser.as_str())
                .map_err(|err| Error::Toml(TomlError::from(err)))
        }
        SerializedType::Json => {
            serde_json::from_reader(file).map_err(Error::Json)
        }
        SerializedType::Encrypted => {
            panic!(
                "We messed up. Somehow we tried to decrypt a file ({:?}) that \
                does not support encryption",
                path.as_ref()
            )
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
    state_path.push(format!("{}", state_id));
    state_path.set_extension("state");

    state_path
}

pub(crate) fn get_data_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut data_path = PathBuf::from(path.as_ref());
    data_path.push("braid");
    data_path.set_extension("data");

    data_path
}

pub(crate) fn get_codebook_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut cb_path = PathBuf::from(path.as_ref());
    cb_path.push("braid");
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
    file_config: FileConfig,
    key: Option<&EncryptionKey>,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let state_path = get_state_path(path, state_id);

    let serialized_type = file_config.serialized_type;

    save_as_possibly_encrypted(
        state,
        state_path.as_path(),
        serialized_type,
        key,
    )?;

    info!("State {} saved to {:?}", state_id, state_path);
    Ok(())
}

/// Save all the states. Assumes the data and codebook exist.
pub(crate) fn save_states<P: AsRef<Path>>(
    path: P,
    states: &[DatalessState],
    state_ids: &[usize],
    file_config: FileConfig,
    key: Option<&EncryptionKey>,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    states
        .iter()
        .zip(state_ids.iter())
        .try_for_each(|(state, id)| {
            save_state(path.as_ref(), state, *id, file_config, key)
        })
}

pub(crate) fn save_data<P: AsRef<Path>>(
    path: P,
    data: &DataStore,
    file_config: FileConfig,
    key: Option<&EncryptionKey>,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let data_path = get_data_path(path);
    save_as_possibly_encrypted(
        data,
        data_path,
        file_config.serialized_type,
        key,
    )
}

pub(crate) fn save_codebook<P: AsRef<Path>>(
    path: P,
    codebook: &Codebook,
    file_config: FileConfig,
    key: Option<&EncryptionKey>,
) -> Result<(), Error> {
    path_validator(path.as_ref())?;
    let cb_path = get_codebook_path(path);
    save_as_possibly_encrypted(
        codebook,
        cb_path,
        file_config.serialized_type,
        key,
    )
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
    file_config: FileConfig,
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

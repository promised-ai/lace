use crate::{Codebook, ColMetadata, ColMetadataList, ColType, RowNameList};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use polars::prelude::{
    CsvReader, DataFrame, DataType, IpcReader, JsonFormat, JsonReader,
    ParquetReader, PolarsError, SerReader, Series,
};
use std::convert::TryFrom;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

pub const DEFAULT_CAT_CUTOFF: u8 = 20;

#[derive(Error, Debug)]
pub enum ReadError {
    #[error("Io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] polars::prelude::PolarsError),
}

pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<DataFrame, ReadError> {
    let mut file = File::open(path)?;
    let df = ParquetReader::new(&mut file).finish()?;
    Ok(df)
}

pub fn read_ipc<P: AsRef<Path>>(path: P) -> Result<DataFrame, ReadError> {
    let mut file = File::open(path)?;
    let df = IpcReader::new(&mut file).finish()?;
    Ok(df)
}

pub fn read_json<P: AsRef<Path>>(path: P) -> Result<DataFrame, ReadError> {
    let ext: String = path.as_ref().extension().map_or_else(
        || String::from(""),
        |ext| ext.to_string_lossy().to_lowercase(),
    );

    let format = match ext.as_str() {
        "json" => JsonFormat::Json,
        "jsonl" => JsonFormat::JsonLines,
        _ => JsonFormat::JsonLines,
    };

    let mut file = File::open(path)?;

    let df = JsonReader::new(&mut file)
        .infer_schema_len(Some(1000))
        .with_json_format(format)
        .finish()?;

    Ok(df)
}

pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<DataFrame, ReadError> {
    let df = CsvReader::from_path(path.as_ref())?
        .infer_schema(Some(1000))
        .has_header(true)
        .finish()?;
    Ok(df)
}

macro_rules! stv_arm {
    ($srs: ident, $method: ident, $X: ty) => {{
        let xs: Vec<$X> = $srs
            .$method()
            .unwrap()
            .into_iter()
            .flat_map(|x_opt| x_opt.map(|x| x as $X))
            .collect();
        xs
    }};
}

macro_rules! series_to_vec {
    ($srs: ident, $X: ty) => {{
        match $srs.dtype() {
            DataType::UInt8 => stv_arm!($srs, u8, $X),
            DataType::UInt16 => stv_arm!($srs, u16, $X),
            DataType::UInt32 => stv_arm!($srs, u32, $X),
            DataType::UInt64 => stv_arm!($srs, u64, $X),
            DataType::Int8 => stv_arm!($srs, i8, $X),
            DataType::Int16 => stv_arm!($srs, i16, $X),
            DataType::Int32 => stv_arm!($srs, i32, $X),
            DataType::Int64 => stv_arm!($srs, i64, $X),
            DataType::Float32 => stv_arm!($srs, f32, $X),
            DataType::Float64 => stv_arm!($srs, f64, $X),
            _ => panic!("unsupported dtype"),
        }
    }};
}

macro_rules! sts_arm {
    ($srs: ident, $method: ident) => {{
        let xs: Vec<String> = $srs
            .$method()
            .unwrap()
            .into_iter()
            .flat_map(|x_opt| x_opt.map(|x| format!("{}", x)))
            .collect();
        xs
    }};
}

macro_rules! series_to_strings {
    ($srs: ident) => {{
        match $srs.dtype() {
            DataType::UInt8 => sts_arm!($srs, u8),
            DataType::UInt16 => sts_arm!($srs, u16),
            DataType::UInt32 => sts_arm!($srs, u32),
            DataType::UInt64 => sts_arm!($srs, u64),
            DataType::Int8 => sts_arm!($srs, i8),
            DataType::Int16 => sts_arm!($srs, i16),
            DataType::Int32 => sts_arm!($srs, i32),
            DataType::Int64 => sts_arm!($srs, i64),
            DataType::Float32 => sts_arm!($srs, f32),
            DataType::Float64 => sts_arm!($srs, f64),
            DataType::Utf8 => sts_arm!($srs, utf8),
            _ => panic!("unsupported dtype"),
        }
    }};
}

fn uint_coltype(srs: &Series, cat_cutoff: Option<u8>) -> ColType {
    let x_max: u64 = srs.max().unwrap();
    let maxval = cat_cutoff.unwrap_or(DEFAULT_CAT_CUTOFF) as u64;
    if x_max >= maxval {
        count_coltype(srs)
    } else {
        uint_categorical_coltype((x_max + 1) as usize)
    }
}

fn int_coltype(srs: &Series, cat_cutoff: Option<u8>) -> ColType {
    let x_min: i64 = srs.min().unwrap();
    if x_min < 0 {
        continuous_coltype(srs)
    } else {
        uint_coltype(srs, cat_cutoff)
    }
}

fn continuous_coltype(srs: &Series) -> ColType {
    let xs: Vec<f64> = series_to_vec!(srs, f64);

    ColType::Continuous {
        hyper: Some(NixHyper::from_data(&xs)),
        prior: None,
    }
}

fn count_coltype(srs: &Series) -> ColType {
    let xs: Vec<u32> = series_to_vec!(srs, u32);
    ColType::Count {
        hyper: Some(PgHyper::from_data(&xs)),
        prior: None,
    }
}

fn uint_categorical_coltype(k: usize) -> ColType {
    ColType::Categorical {
        k,
        hyper: Some(CsdHyper::new(1.0, 1.0)),
        prior: None,
        value_map: None,
    }
}

fn string_categorical_coltype(srs: &Series) -> Result<ColType, PolarsError> {
    use std::collections::BTreeSet;

    let n_unique = srs.n_unique()?;
    if n_unique >= std::u8::MAX as usize {
        panic!("Too many categories")
    } else {
        let unique: BTreeSet<String> = srs
            .unique()?
            .utf8()?
            .into_iter()
            .filter_map(|x| x.map(String::from))
            .collect();

        let value_map = unique.iter().cloned().enumerate().collect();

        Ok(ColType::Categorical {
            k: n_unique,
            hyper: Some(CsdHyper::new(1.0, 1.0)),
            prior: None,
            value_map: Some(value_map),
        })
    }
}

fn series_to_colmd(srs: &Series, cat_cutoff: Option<u8>) -> ColMetadata {
    let name = String::from(srs.name());
    let dtype = srs.dtype();
    let coltype = match dtype {
        DataType::Boolean => uint_categorical_coltype(2),
        DataType::UInt8 => uint_coltype(srs, cat_cutoff),
        DataType::UInt16 => uint_coltype(srs, cat_cutoff),
        DataType::UInt32 => uint_coltype(srs, cat_cutoff),
        DataType::UInt64 => uint_coltype(srs, cat_cutoff),
        DataType::Int8 => int_coltype(srs, cat_cutoff),
        DataType::Int16 => int_coltype(srs, cat_cutoff),
        DataType::Int32 => int_coltype(srs, cat_cutoff),
        DataType::Int64 => int_coltype(srs, cat_cutoff),
        DataType::Float32 => continuous_coltype(srs),
        DataType::Float64 => continuous_coltype(srs),
        DataType::Utf8 => string_categorical_coltype(srs).unwrap(),
        // DataType::Categorical(mapping_opt) => {}
        // DataType::Null => {}
        // DataType::Unknown => {}
        _ => panic!("unsupported data type: {}", dtype),
    };

    ColMetadata {
        name,
        coltype,
        notes: None,
    }
}

pub fn df_to_codebook(
    df: DataFrame,
    cat_cutoff: Option<u8>,
    alpha_prior_opt: Option<CrpPrior>,
) -> Codebook {
    let mut columns = df.get_columns().iter();

    // FIXME: make sure the that the first column is ID
    let row_names = {
        let id_col: &Series = columns.next().expect("Empty dataframe?");
        assert_eq!(id_col.name().to_lowercase(), "id");
        assert_eq!(id_col.null_count(), 0);
        let indices: Vec<String> = series_to_strings!(id_col);
        RowNameList::try_from(indices).unwrap()
    };

    let col_metadata = {
        let col_metadata = columns
            .map(|srs| series_to_colmd(srs, cat_cutoff))
            .collect::<Vec<_>>();

        ColMetadataList::try_from(col_metadata).unwrap()
    };

    let alpha_prior = alpha_prior_opt
        .unwrap_or_else(|| braid_consts::general_alpha_prior().into());

    Codebook {
        table_name: "my_table".into(),
        state_alpha_prior: Some(alpha_prior.clone()),
        view_alpha_prior: Some(alpha_prior),
        col_metadata,
        row_names,
        comments: None,
    }
}

macro_rules! codebook_from_fn {
    ($fn_name: ident, $reader: ident) => {
        pub fn $fn_name<P: AsRef<Path>>(
            path: P,
            cat_cutoff: Option<u8>,
            alpha_prior_opt: Option<CrpPrior>,
        ) -> Codebook {
            let df = $reader(path).unwrap();
            df_to_codebook(df, cat_cutoff, alpha_prior_opt)
        }
    };
}

codebook_from_fn!(codebook_from_csv, read_csv);
codebook_from_fn!(codebook_from_parquet, read_parquet);
codebook_from_fn!(codebook_from_ipc, read_ipc);
codebook_from_fn!(codebook_from_json, read_json);

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn tempfile() -> NamedTempFile {
        NamedTempFile::new().unwrap()
    }

    fn write_to_tempfile(s: &str) -> NamedTempFile {
        let mut file = tempfile();
        file.write_all(s.as_bytes()).unwrap();
        file
    }

    #[test]
    fn codebook_with_all_types_inferse_correct_types_csv() {
        let data = "\
            id,cat_str,cat_int,count,cts_int,cts_float
            0,       A,      1,    0,    -1,      1.0
            1,        ,      0,  256,     0,      2.0
            2,       B,      1,    2,    12,      3.0
            3,       A,      1,     ,      ,
            4,       A,       ,   43,     3,\
        "
        .replace(' ', "");

        let file = write_to_tempfile(&data);

        let codebook = codebook_from_csv(file.path(), None, None);

        assert_eq!(codebook.col_metadata.len(), 5);
        assert_eq!(codebook.row_names.len(), 5);

        let cat_str = codebook.col_metadata.get("cat_str").unwrap().1;
        let cat_int = codebook.col_metadata.get("cat_int").unwrap().1;
        let count = codebook.col_metadata.get("count").unwrap().1;
        let cts_int = codebook.col_metadata.get("cts_int").unwrap().1;
        let cts_float = codebook.col_metadata.get("cts_float").unwrap().1;

        assert!(cat_str.coltype.is_categorical());
        assert!(cat_int.coltype.is_categorical());
        assert!(count.coltype.is_count());
        assert!(cts_int.coltype.is_continuous());
        assert!(cts_float.coltype.is_continuous());
    }

    #[test]
    fn codebook_with_all_types_inferse_correct_types_json() {
        let data = r#"
            {"id" : 0, "cat_str": "A", "cat_int": 1, "count": 0, "cts_int": -1, "cts_float": 1.0}
            {"id" : 1, "cat_int": 0, "count": 256, "cts_int": 0, "cts_float": 2.0}
            {"id" : 2, "cat_str": "B", "cat_int": 1, "count": 2, "cts_int": 12, "cts_float": 3.0}
            {"id" : 3, "cat_str": "A", "cat_int": 1 }
            {"id" : 4, "cat_str": "A", "count": 43, "cts_int": 3}
        "#.to_string().replace(' ', "");

        let file = write_to_tempfile(data.trim());

        let codebook = codebook_from_json(file.path(), None, None);

        assert_eq!(codebook.col_metadata.len(), 5);
        assert_eq!(codebook.row_names.len(), 5);

        let cat_str = codebook.col_metadata.get("cat_str").unwrap().1;
        let cat_int = codebook.col_metadata.get("cat_int").unwrap().1;
        let count = codebook.col_metadata.get("count").unwrap().1;
        let cts_int = codebook.col_metadata.get("cts_int").unwrap().1;
        let cts_float = codebook.col_metadata.get("cts_float").unwrap().1;

        assert!(cat_str.coltype.is_categorical());
        assert!(cat_int.coltype.is_categorical());
        assert!(count.coltype.is_count());
        assert!(cts_int.coltype.is_continuous());
        assert!(cts_float.coltype.is_continuous());
    }
}

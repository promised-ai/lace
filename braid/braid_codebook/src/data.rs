use crate::error::{CodebookError, ReadError};
use crate::{Codebook, ColMetadata, ColMetadataList, ColType, RowNameList};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use polars::prelude::{
    CsvReader, DataFrame, DataType, IpcReader, JsonFormat, JsonReader,
    ParquetReader, SerReader, Series,
};
use std::convert::TryFrom;
use std::fs::File;
use std::path::Path;

pub const DEFAULT_CAT_CUTOFF: u8 = 20;

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

#[macro_export]
macro_rules! series_to_opt_vec {
    ($srs: ident, $X: ty) => {{
        macro_rules! stv_arm {
            ($srsi: ident, $method: ident, $Xi: ty) => {{
                $srsi
                    .$method()?
                    .into_iter()
                    .map(|x_opt| x_opt.map(|x| x as $Xi))
            }};
        }
        match $srs.dtype() {
            polars::prelude::DataType::UInt8 => {
                stv_arm!($srs, u8, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::UInt16 => {
                stv_arm!($srs, u16, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::UInt32 => {
                stv_arm!($srs, u32, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::UInt64 => {
                stv_arm!($srs, u64, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::Int8 => {
                stv_arm!($srs, i8, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::Int16 => {
                stv_arm!($srs, i16, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::Int32 => {
                stv_arm!($srs, i32, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::Int64 => {
                stv_arm!($srs, i64, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::Float32 => {
                stv_arm!($srs, f32, $X).collect::<Vec<Option<$X>>>()
            }
            polars::prelude::DataType::Float64 => {
                stv_arm!($srs, f64, $X).collect::<Vec<Option<$X>>>()
            }
            _ => {
                return Err($crate::CodebookError::UnableToInferColumnType {
                    col_name: $srs.name().to_owned(),
                })
            }
        }
    }};
}

#[macro_export]
macro_rules! series_to_vec {
    ($srs: ident, $X: ty) => {{
        macro_rules! stv_arm {
            ($srsi: ident, $method: ident, $Xi: ty) => {{
                $srsi
                    .$method()?
                    .into_iter()
                    .map(|x_opt| x_opt.map(|x| x as $Xi))
            }};
        }
        match $srs.dtype() {
            polars::prelude::DataType::UInt8 => {
                stv_arm!($srs, u8, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::UInt16 => {
                stv_arm!($srs, u16, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::UInt32 => {
                stv_arm!($srs, u32, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::UInt64 => {
                stv_arm!($srs, u64, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::Int8 => {
                stv_arm!($srs, i8, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::Int16 => {
                stv_arm!($srs, i16, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::Int32 => {
                stv_arm!($srs, i32, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::Int64 => {
                stv_arm!($srs, i64, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::Float32 => {
                stv_arm!($srs, f32, $X).flatten().collect::<Vec<$X>>()
            }
            polars::prelude::DataType::Float64 => {
                stv_arm!($srs, f64, $X).flatten().collect::<Vec<$X>>()
            }
            _ => {
                return Err($crate::CodebookError::UnableToInferColumnType {
                    col_name: $srs.name().to_owned(),
                })
            }
        }
    }};
}

#[macro_export]
macro_rules! series_to_opt_strings {
    ($srs: ident) => {{
        macro_rules! sts_arm {
            ($srsi: ident, $method: ident) => {{
                $srsi
                    .$method()?
                    .into_iter()
                    .map(|x_opt| x_opt.map(|x| format!("{}", x)))
            }};
        }
        match $srs.dtype() {
            polars::prelude::DataType::UInt8 => {
                sts_arm!($srs, u8).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::UInt16 => {
                sts_arm!($srs, u16).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::UInt32 => {
                sts_arm!($srs, u32).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::UInt64 => {
                sts_arm!($srs, u64).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Int8 => {
                sts_arm!($srs, i8).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Int16 => {
                sts_arm!($srs, i16).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Int32 => {
                sts_arm!($srs, i32).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Int64 => {
                sts_arm!($srs, i64).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Float32 => {
                sts_arm!($srs, f32).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Float64 => {
                sts_arm!($srs, f64).collect::<Vec<Option<String>>>()
            }
            polars::prelude::DataType::Utf8 => {
                sts_arm!($srs, utf8).collect::<Vec<Option<String>>>()
            }
            _ => {
                return Err($crate::CodebookError::UnableToInferColumnType {
                    col_name: $srs.name().to_owned(),
                })
            }
        }
    }};
}

#[macro_export]
macro_rules! series_to_strings {
    ($srs: ident) => {{
        macro_rules! sts_arm {
            ($srsi: ident, $method: ident) => {{
                $srsi
                    .$method()?
                    .into_iter()
                    .map(|x_opt| x_opt.map(|x| format!("{}", x)))
            }};
        }
        match $srs.dtype() {
            polars::prelude::DataType::UInt8 => {
                sts_arm!($srs, u8).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::UInt16 => {
                sts_arm!($srs, u16).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::UInt32 => {
                sts_arm!($srs, u32).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::UInt64 => {
                sts_arm!($srs, u64).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Int8 => {
                sts_arm!($srs, i8).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Int16 => {
                sts_arm!($srs, i16).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Int32 => {
                sts_arm!($srs, i32).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Int64 => {
                sts_arm!($srs, i64).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Float32 => {
                sts_arm!($srs, f32).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Float64 => {
                sts_arm!($srs, f64).flatten().collect::<Vec<String>>()
            }
            polars::prelude::DataType::Utf8 => {
                sts_arm!($srs, utf8).flatten().collect::<Vec<String>>()
            }
            _ => {
                return Err($crate::CodebookError::UnableToInferColumnType {
                    col_name: $srs.name().to_owned(),
                })
            }
        }
    }};
}

pub use series_to_opt_strings;
pub use series_to_opt_vec;
pub use series_to_strings;
pub use series_to_vec;

fn uint_coltype(
    srs: &Series,
    cat_cutoff: Option<u8>,
    no_hypers: bool,
) -> Result<ColType, CodebookError> {
    let x_max: u64 = srs.max().unwrap();
    let maxval = cat_cutoff.unwrap_or(DEFAULT_CAT_CUTOFF) as u64;
    if x_max >= maxval {
        count_coltype(srs, no_hypers)
    } else {
        uint_categorical_coltype((x_max + 1) as usize, no_hypers)
    }
}

fn int_coltype(
    srs: &Series,
    cat_cutoff: Option<u8>,
    no_hypers: bool,
) -> Result<ColType, CodebookError> {
    let x_min: i64 = srs.min().unwrap();
    if x_min < 0 {
        continuous_coltype(srs, no_hypers)
    } else {
        uint_coltype(srs, cat_cutoff, no_hypers)
    }
}

fn continuous_coltype(
    srs: &Series,
    no_hypers: bool,
) -> Result<ColType, CodebookError> {
    let xs: Vec<f64> = series_to_vec!(srs, f64);

    let (hyper, prior) = if no_hypers {
        (None, Some(braid_stats::prior::nix::from_data(&xs)))
    } else {
        (Some(NixHyper::from_data(&xs)), None)
    };

    Ok(ColType::Continuous { hyper, prior })
}

fn count_coltype(
    srs: &Series,
    no_hypers: bool,
) -> Result<ColType, CodebookError> {
    let xs: Vec<u32> = series_to_vec!(srs, u32);

    let (hyper, prior) = if no_hypers {
        (None, Some(braid_stats::prior::pg::from_data(&xs)))
    } else {
        (Some(PgHyper::from_data(&xs)), None)
    };

    Ok(ColType::Count { hyper, prior })
}

fn uint_categorical_coltype(
    k: usize,
    no_hypers: bool,
) -> Result<ColType, CodebookError> {
    use braid_stats::rv::dist::SymmetricDirichlet;

    let (hyper, prior) = if no_hypers {
        (None, Some(SymmetricDirichlet::jeffreys(k).unwrap()))
    } else {
        (Some(CsdHyper::new(1.0, 1.0)), None)
    };

    Ok(ColType::Categorical {
        k,
        hyper,
        prior,
        value_map: None,
    })
}

fn string_categorical_coltype(
    srs: &Series,
    no_hypers: bool,
) -> Result<ColType, CodebookError> {
    use braid_stats::rv::dist::SymmetricDirichlet;
    use std::collections::{BTreeMap, BTreeSet};

    let n_unique = srs.n_unique()?;
    if n_unique > std::u8::MAX as usize {
        Err(CodebookError::CategoricalOverflow {
            col_name: srs.name().to_owned(),
        })
    } else {
        let unique: BTreeSet<String> = srs
            .unique()?
            .utf8()?
            .into_iter()
            .filter_map(|x| x.map(String::from))
            .collect();

        let value_map: BTreeMap<usize, String> =
            unique.iter().cloned().enumerate().collect();

        let n_null = srs.null_count();
        let k = n_unique - (n_null > 0) as usize;

        assert_eq!(
            k,
            value_map.len(),
            "Number of unique values in categorical columns does not match the \
            length of the value map"
        );

        let (hyper, prior) = if no_hypers {
            (None, Some(SymmetricDirichlet::jeffreys(k).unwrap()))
        } else {
            (Some(CsdHyper::new(1.0, 1.0)), None)
        };

        Ok(ColType::Categorical {
            k,
            hyper,
            prior,
            value_map: Some(value_map),
        })
    }
}

fn series_to_colmd(
    srs: &Series,
    cat_cutoff: Option<u8>,
    no_hypers: bool,
) -> Result<ColMetadata, CodebookError> {
    let name = String::from(srs.name());
    let dtype = srs.dtype();
    let coltype = match dtype {
        DataType::Boolean => uint_categorical_coltype(2, no_hypers),
        DataType::UInt8 => uint_coltype(srs, cat_cutoff, no_hypers),
        DataType::UInt16 => uint_coltype(srs, cat_cutoff, no_hypers),
        DataType::UInt32 => uint_coltype(srs, cat_cutoff, no_hypers),
        DataType::UInt64 => uint_coltype(srs, cat_cutoff, no_hypers),
        DataType::Int8 => int_coltype(srs, cat_cutoff, no_hypers),
        DataType::Int16 => int_coltype(srs, cat_cutoff, no_hypers),
        DataType::Int32 => int_coltype(srs, cat_cutoff, no_hypers),
        DataType::Int64 => int_coltype(srs, cat_cutoff, no_hypers),
        DataType::Float32 => continuous_coltype(srs, no_hypers),
        DataType::Float64 => continuous_coltype(srs, no_hypers),
        DataType::Utf8 => string_categorical_coltype(srs, no_hypers),
        DataType::Null => Err(CodebookError::BlankColumn {
            col_name: name.clone(),
        }),
        _ => Err(CodebookError::UnsupportedDataType {
            col_name: name.clone(),
            dtype: dtype.clone(),
        }),
        // DataType::Categorical(mapping_opt) => {}
        // DataType::Unknown => {}
    }?;

    Ok(ColMetadata {
        name,
        coltype,
        notes: None,
        missing_not_at_random: false,
    })
}

fn rownames_from_index(id_srs: &Series) -> Result<RowNameList, CodebookError> {
    // this should not be able to happen due to user error, so we panic
    assert_eq!(id_srs.name().to_lowercase(), "id");

    if id_srs.null_count() > 0 {
        return Err(CodebookError::NullValuesInIndex);
    }

    let indices: Vec<String> = series_to_strings!(id_srs);
    let row_names = RowNameList::try_from(indices)?;
    Ok(row_names)
}

pub fn df_to_codebook(
    df: &DataFrame,
    cat_cutoff: Option<u8>,
    alpha_prior_opt: Option<CrpPrior>,
    no_hypers: bool,
) -> Result<Codebook, CodebookError> {
    let (col_metadata, row_names) = {
        let mut row_names_opt: Option<RowNameList> = None;
        let mut col_metadata = Vec::with_capacity(df.shape().1);
        for srs in df.get_columns().iter() {
            if srs.name().to_lowercase() == "id" {
                if row_names_opt.is_some() {
                    return Err(CodebookError::MultipleIdColumns);
                }
                row_names_opt = Some(rownames_from_index(srs)?);
            } else {
                if srs.n_unique()? < 2 {
                    return Err(CodebookError::SingleValueColumn(
                        srs.name().to_owned(),
                    ));
                }
                let colmd = series_to_colmd(srs, cat_cutoff, no_hypers)?;
                col_metadata.push(colmd);
            }
        }

        let row_names = row_names_opt.ok_or(CodebookError::NoIdColumn)?;
        let col_metadata = ColMetadataList::try_from(col_metadata)?;

        (col_metadata, row_names)
    };

    let alpha_prior = alpha_prior_opt
        .unwrap_or_else(|| braid_consts::general_alpha_prior().into());

    Ok(Codebook {
        table_name: "my_table".into(),
        state_alpha_prior: Some(alpha_prior.clone()),
        view_alpha_prior: Some(alpha_prior),
        col_metadata,
        row_names,
        comments: None,
    })
}

macro_rules! codebook_from_fn {
    ($fn_name: ident, $reader: ident) => {
        pub fn $fn_name<P: AsRef<Path>>(
            path: P,
            cat_cutoff: Option<u8>,
            alpha_prior_opt: Option<braid_stats::prior::crp::CrpPrior>,
            no_hypers: bool,
        ) -> Result<$crate::Codebook, $crate::CodebookError> {
            let df = $reader(path).unwrap();
            df_to_codebook(&df, cat_cutoff, alpha_prior_opt, no_hypers)
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
    use polars::prelude::*;
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

        let codebook =
            codebook_from_csv(file.path(), None, None, false).unwrap();

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

        let codebook =
            codebook_from_json(file.path(), None, None, false).unwrap();

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
    fn string_col_value_map_should_be_sorted_no_null() {
        let srs = Series::new("a", vec!["dog", "cat", "bear", "fox"]);
        let coltype = string_categorical_coltype(&srs, true).unwrap();
        match coltype {
            ColType::Categorical {
                k,
                value_map: Some(value_map),
                ..
            } => {
                assert_eq!(k, value_map.len());
                assert_eq!(value_map.get(&0).unwrap(), "bear");
                assert_eq!(value_map.get(&1).unwrap(), "cat");
                assert_eq!(value_map.get(&2).unwrap(), "dog");
                assert_eq!(value_map.get(&3).unwrap(), "fox");
            }
            ColType::Categorical {
                value_map: None, ..
            } => {
                panic!("No value map")
            }
            _ => panic!("wrong coltype"),
        }
    }

    #[test]
    fn string_col_value_map_should_be_sorted_null() {
        let srs = Series::new(
            "a",
            vec![Some("dog"), Some("cat"), None, Some("bear"), Some("fox")],
        );
        let coltype = string_categorical_coltype(&srs, true).unwrap();
        match coltype {
            ColType::Categorical {
                k,
                value_map: Some(value_map),
                ..
            } => {
                assert_eq!(k, value_map.len());
                assert_eq!(value_map.get(&0).unwrap(), "bear");
                assert_eq!(value_map.get(&1).unwrap(), "cat");
                assert_eq!(value_map.get(&2).unwrap(), "dog");
                assert_eq!(value_map.get(&3).unwrap(), "fox");
            }
            ColType::Categorical {
                value_map: None, ..
            } => {
                panic!("No value map")
            }
            _ => panic!("wrong coltype"),
        }
    }

    mod inference {
        use super::*;

        macro_rules! categorical_or_count {
            ($test_name: ident, $int_min: expr, $int_max: expr, $cat_cutoff: expr, $should_be_categorical: expr) => {
                #[test]
                fn $test_name() {
                    let srs = Series::new(
                        "a",
                        ($int_min..$int_max).collect::<Vec<_>>(),
                    );
                    let colmd =
                        series_to_colmd(&srs, $cat_cutoff, true).unwrap();
                    match colmd.coltype {
                        ColType::Categorical {
                            k, value_map: None, ..
                        } => {
                            if $should_be_categorical {
                                assert_eq!(k, $int_max as usize);
                            } else {
                                panic!("should not be categorical");
                            }
                        }
                        ColType::Categorical {
                            value_map: Some(_), ..
                        } => {
                            panic!("int categorical should not have value_map")
                        }
                        _ => {
                            if $should_be_categorical {
                                panic!("should have been categorical");
                            }
                        }
                    }
                }
            };
        }

        categorical_or_count!(u64_1, 0_u64, 20, Some(20), true);
        categorical_or_count!(u64_2, 0_u64, 21, Some(20), false);

        categorical_or_count!(u32_1, 0_u32, 20, Some(20), true);
        categorical_or_count!(u32_2, 0_u32, 21, Some(20), false);

        categorical_or_count!(u16_1, 0_u16, 20, Some(20), true);
        categorical_or_count!(u16_2, 0_u16, 21, Some(20), false);

        categorical_or_count!(u8_1, 0_u8, 20, Some(20), true);
        categorical_or_count!(u8_2, 0_u8, 21, Some(20), false);

        categorical_or_count!(i8_1, 0_i8, 20, Some(20), true);
        categorical_or_count!(i8_2, 0_i8, 21, Some(20), false);
        categorical_or_count!(i8_3, -1_i8, 10, Some(20), false);
        categorical_or_count!(i8_4, -1_i8, 21, Some(20), false);

        categorical_or_count!(i16_1, 0_i16, 20, Some(20), true);
        categorical_or_count!(i16_2, 0_i16, 21, Some(20), false);
        categorical_or_count!(i16_3, -1_i16, 10, Some(20), false);
        categorical_or_count!(i16_4, -1_i16, 21, Some(20), false);

        categorical_or_count!(i32_1, 0_i32, 20, Some(20), true);
        categorical_or_count!(i32_2, 0_i32, 21, Some(20), false);
        categorical_or_count!(i32_3, -1_i32, 10, Some(20), false);
        categorical_or_count!(i32_4, -1_i32, 21, Some(20), false);

        categorical_or_count!(i64_1, 0_i64, 20, Some(20), true);
        categorical_or_count!(i64_2, 0_i64, 21, Some(20), false);
        categorical_or_count!(i64_3, -1_i64, 10, Some(20), false);
        categorical_or_count!(i64_4, -1_i64, 21, Some(20), false);

        #[test]
        fn greater_than_256_string_values_errors() {
            let srs = Series::new(
                "A",
                (0..256).map(|x| format!("{x}")).collect::<Vec<_>>(),
            );
            match series_to_colmd(&srs, None, false) {
                Err(CodebookError::CategoricalOverflow { .. }) => {}
                Err(err) => panic!("wrong error: {}", err),
                Ok(_) => panic!("should have failed"),
            }
        }

        #[test]
        fn exactly_255_string_values_ok() {
            let srs = Series::new(
                "A",
                (0..255).map(|x| format!("{x}")).collect::<Vec<_>>(),
            );
            assert!(series_to_colmd(&srs, None, false).is_ok());
        }

        #[test]
        fn fewer_than_255_string_values_ok() {
            let srs = Series::new(
                "A",
                (0..25)
                    .cycle()
                    .take(100)
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>(),
            );
            assert!(series_to_colmd(&srs, None, false).is_ok());
        }

        macro_rules! count_or_continuous {
            ($test_name: ident, $min_val: expr, $max_val: expr, $is_count: expr) => {
                #[test]
                fn $test_name() {
                    let srs = Series::new(
                        "A",
                        ($min_val..$max_val).collect::<Vec<_>>(),
                    );
                    let colmd = series_to_colmd(&srs, None, false).unwrap();
                    match colmd.coltype {
                        ColType::Continuous { .. } => assert!(!$is_count),
                        ColType::Count { .. } => assert!($is_count),
                        _ => panic!("Unexpected col type"),
                    };
                }
            };
        }

        count_or_continuous!(count_or_cts_u16, 1_u16, 300, true);
        count_or_continuous!(count_or_cts_u32, 1_u32, 300, true);
        count_or_continuous!(count_or_cts_u64, 1_u64, 300, true);
        count_or_continuous!(count_or_cts_i16, 1_i16, 300, true);
        count_or_continuous!(count_or_cts_i32, 1_i32, 300, true);
        count_or_continuous!(count_or_cts_i64, 1_i64, 300, true);

        count_or_continuous!(count_or_cts_i16_neg, -1_i16, 300, false);
        count_or_continuous!(count_or_cts_i32_neg, -1_i32, 300, false);
        count_or_continuous!(count_or_cts_i64_neg, -1_i64, 300, false);

        count_or_continuous!(count_or_cts_i16_neg_small, -1_i16, 10, false);
        count_or_continuous!(count_or_cts_i32_neg_small, -1_i32, 10, false);
        count_or_continuous!(count_or_cts_i64_neg_small, -1_i64, 10, false);

        #[test]
        fn bool_data_is_categorical() {
            let srs = Series::new(
                "A",
                (0..100).map(|x| x % 2 == 1).collect::<Vec<bool>>(),
            );
            let colmd = series_to_colmd(&srs, None, true).unwrap();
            match colmd.coltype {
                ColType::Categorical {
                    value_map: None, ..
                } => (),
                ColType::Categorical {
                    value_map: Some(_), ..
                } => {
                    panic!("value map should be none")
                }
                _ => panic!("wrong coltype"),
            }
        }
    }
}

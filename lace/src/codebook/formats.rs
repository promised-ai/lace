use std::fs::File;
use std::num::NonZero;
use std::path::Path;

use polars::prelude::CsvReadOptions;
use polars::prelude::DataFrame;
use polars::prelude::IpcReader;
use polars::prelude::JsonFormat;
use polars::prelude::JsonReader;
use polars::prelude::ParquetReader;
use polars::prelude::SerReader;

use super::ReadError;

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
        .infer_schema_len(NonZero::new(1000))
        .with_json_format(format)
        .finish()?;

    Ok(df)
}

pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<DataFrame, ReadError> {
    let df = CsvReadOptions::default()
        .with_infer_schema_length(Some(1000))
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.as_ref().into()))?
        .finish()?;
    Ok(df)
}

macro_rules! codebook_from_fn {
    ($fn_name: ident, $reader: ident) => {
        pub fn $fn_name<P: AsRef<Path>>(
            path: P,
            cat_cutoff: Option<u32>,
            state_prior_process: Option<$crate::codebook::PriorProcess>,
            view_prior_process: Option<$crate::codebook::PriorProcess>,
            no_hypers: bool,
        ) -> Result<
            $crate::codebook::Codebook,
            $crate::codebook::error::CodebookError,
        > {
            let df = $reader(path).unwrap();
            $crate::codebook::data::df_to_codebook(
                &df,
                cat_cutoff,
                state_prior_process,
                view_prior_process,
                no_hypers,
            )
        }
    };
}

codebook_from_fn!(codebook_from_csv, read_csv);
codebook_from_fn!(codebook_from_parquet, read_parquet);
codebook_from_fn!(codebook_from_ipc, read_ipc);
codebook_from_fn!(codebook_from_json, read_json);

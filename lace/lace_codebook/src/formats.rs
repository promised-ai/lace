use crate::ReadError;
use polars::prelude::{
    CsvReader, DataFrame, IpcReader, JsonFormat, JsonReader, ParquetReader,
    SerReader,
};
use std::fs::File;
use std::path::Path;

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

macro_rules! codebook_from_fn {
    ($fn_name: ident, $reader: ident) => {
        pub fn $fn_name<P: AsRef<Path>>(
            path: P,
            cat_cutoff: Option<u8>,
            state_prior_process: Option<$crate::codebook::PriorProcess>,
            view_prior_process: Option<$crate::codebook::PriorProcess>,
            no_hypers: bool,
        ) -> Result<$crate::codebook::Codebook, $crate::error::CodebookError> {
            let df = $reader(path).unwrap();
            $crate::data::df_to_codebook(
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

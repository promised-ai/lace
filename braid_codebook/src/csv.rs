//! Utilities to generate codebooks from CSV files
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::convert::{From, TryInto};
use std::f64;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{Cursor, Read};
use std::mem::transmute_copy;
use std::path::PathBuf;
use std::str::FromStr;

use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use braid_utils::UniqueCollection;
use csv::{Reader, ReaderBuilder};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use thiserror::Error;

use crate::codebook::{
    Codebook, ColMetadata, ColMetadataList, ColType, RowNameList,
};

/// Errors that can arise when creating a codebook from a CSV file
#[derive(Debug, Error)]
pub enum FromCsvError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("csv error: {0}")]
    CsvError(#[from] csv::Error),
    /// A column had no values in it.
    #[error("column '{col_name}' is blank")]
    BlankColumn { col_name: String },
    /// Could not infer the data type of a column
    #[error("cannot infer feature type of column '{col_name}'")]
    UnableToInferColumnType { col_name: String },
    /// Too many distinct values for categorical column. A category is
    /// represented by a u8, so there can only be 256 distinct values.
    #[error("column '{col_name}' contains more than 256 categorical classes")]
    CategoricalOverflow { col_name: String },
    /// The column with name appears more than once
    #[error("column '{col_name}' appears more than once")]
    DuplicatColumn { col_name: String },
    /// The column with name appears more than once
    #[error("row '{row_name}' appears more than once")]
    DuplicatRow { row_name: String },
}

// The type of entry in the CSV cell. Currently Int only supports u8 because
// `categorical` is the only integer type.
#[derive(Clone, Debug, PartialOrd)]
enum Entry {
    SmallUInt(u8),
    UInt(u32),
    Int(i32),
    Float(f64),
    Other(String),
    EmptyCell,
}

impl Eq for Entry {}

macro_rules! entry_eq_arm {
    ($variant:ident, $other:ident, $x: ident) => {
        if let Entry::$variant(y) = $other {
            $x == y
        } else {
            false
        }
    };
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Entry::Float(x) => {
                if let Entry::Float(y) = other {
                    unsafe {
                        // this covers NaNs
                        let xc: u64 = transmute_copy(x);
                        let yc: u64 = transmute_copy(y);
                        xc == yc
                    }
                } else {
                    false
                }
            }
            Entry::Int(x) => entry_eq_arm!(Int, other, x),
            Entry::UInt(x) => entry_eq_arm!(UInt, other, x),
            Entry::SmallUInt(x) => entry_eq_arm!(SmallUInt, other, x),
            Entry::Other(x) => entry_eq_arm!(Other, other, x),
            Entry::EmptyCell => matches!(other, Entry::EmptyCell),
        }
    }
}

impl Hash for Entry {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Entry::Float(x) => unsafe {
                let y: u64 = transmute_copy(x);
                y.hash(state)
            },
            Entry::Int(x) => x.hash(state),
            Entry::UInt(x) => x.hash(state),
            Entry::SmallUInt(x) => x.hash(state),
            Entry::Other(x) => x.hash(state),
            Entry::EmptyCell => 0_u8.hash(state),
        }
    }
}

impl From<&str> for Entry {
    fn from(val: &str) -> Entry {
        let s = val.trim();
        if s.is_empty() {
            return Entry::EmptyCell;
        }

        // preference:
        // 1. SmallUInt
        // 2. UInt
        // 3. Int
        // 4. Float
        // 5. Other
        u8::from_str(s)
            .map(Entry::SmallUInt)
            .or_else(|_| u32::from_str(s).map(Entry::UInt))
            .or_else(|_| i32::from_str(s).map(Entry::Int))
            .or_else(|_| f64::from_str(s).map(Entry::Float))
            .unwrap_or_else(|_| Entry::Other(s.to_owned()))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EntryConversionError {
    EmptyCell,
    InvalidInnerType,
}

macro_rules! impl_try_into_entry {
    ($type: ty, $variant: ident) => {
        impl TryInto<$type> for Entry {
            type Error = EntryConversionError;

            fn try_into(self) -> Result<$type, Self::Error> {
                match self {
                    Entry::$variant(inner) => Ok(inner.into()),
                    Entry::EmptyCell => Err(EntryConversionError::EmptyCell),
                    _ => Err(EntryConversionError::InvalidInnerType),
                }
            }
        }
    };
}

// Depending on the categorical cutoff, both Int and Float types may end up in
// a continuous column, so we'll need to convert both types to f64.
impl TryInto<f64> for Entry {
    type Error = EntryConversionError;

    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            Entry::Float(inner) => Ok(inner),
            Entry::Int(inner) => Ok(f64::from(inner)),
            Entry::UInt(inner) => Ok(f64::from(inner)),
            Entry::SmallUInt(inner) => Ok(f64::from(inner)),
            Entry::EmptyCell => Err(EntryConversionError::EmptyCell),
            _ => Err(EntryConversionError::InvalidInnerType),
        }
    }
}

impl TryInto<u32> for Entry {
    type Error = EntryConversionError;

    fn try_into(self) -> Result<u32, Self::Error> {
        match self {
            Entry::UInt(inner) => Ok(inner),
            Entry::SmallUInt(inner) => Ok(u32::from(inner)),
            Entry::EmptyCell => Err(EntryConversionError::EmptyCell),
            _ => Err(EntryConversionError::InvalidInnerType),
        }
    }
}

impl_try_into_entry!(u8, SmallUInt);
impl_try_into_entry!(i32, Int);
impl_try_into_entry!(String, Other);

#[derive(Clone, Debug, PartialEq, Eq)]
enum ColumnType {
    Categorical,
    Continuous,
    Count,
    Unknown,
    Blank,
}

#[derive(Clone, Debug)]
struct EntryTally {
    pub n: usize,
    pub n_float: usize,
    pub n_int: usize,
    pub n_uint: usize,
    pub n_small_uint: usize,
    pub n_other: usize,
    pub n_empty: usize,
}

impl EntryTally {
    fn new() -> EntryTally {
        EntryTally {
            n: 0,
            n_float: 0,
            n_int: 0,
            n_uint: 0,
            n_small_uint: 0,
            n_other: 0,
            n_empty: 0,
        }
    }

    fn incr(&mut self, entry: &Entry) {
        self.n += 1;
        match entry {
            Entry::SmallUInt(..) => self.n_small_uint += 1,
            Entry::Float(..) => self.n_float += 1,
            Entry::Other(..) => self.n_other += 1,
            Entry::EmptyCell => self.n_empty += 1,
            Entry::Int(..) => self.n_int += 1,
            Entry::UInt(..) => self.n_uint += 1,
        }
    }

    fn n_int_type(&self) -> usize {
        self.n_int + self.n_uint + self.n_small_uint
    }

    fn column_type(&self, col: &[Entry], cat_cutoff: usize) -> ColumnType {
        if self.n_empty == self.n {
            // If all are blank => Blank
            ColumnType::Blank
        } else if self.n_small_uint + self.n_empty == self.n {
            // If all are unsigned small integers and there are fewer unique
            // values than cat_cutoff => Categorical, otherwise => Count
            let n_unique = col.n_unique_cutoff(cat_cutoff);
            if n_unique < cat_cutoff {
                ColumnType::Categorical
            } else {
                ColumnType::Count
            }
        } else if self.n_small_uint + self.n_uint + self.n_empty == self.n {
            // If all values are small or large uint, or missing => Count
            ColumnType::Count
        } else if self.n_float + self.n_int_type() + self.n_empty == self.n {
            // If all values are int type, float, or misggin => Continuous
            ColumnType::Continuous
        } else if self.n_other > 0 {
            // If all values are other (string) or missing and the number of
            // unique values is less than or equal to the number of classes
            // supported by u8 (255).
            let n_unique = col.n_unique_cutoff(256);
            if n_unique < 256 {
                ColumnType::Categorical
            } else {
                ColumnType::Unknown
            }
        } else {
            ColumnType::Unknown
        }
    }
}

fn column_to_categorical_coltype(
    col: Vec<Entry>,
    tally: &EntryTally,
    col_name: &str,
) -> Result<ColType, FromCsvError> {
    use braid_stats::prior::csd::CsdHyper;

    if tally.n == tally.n_small_uint + tally.n_empty {
        // Assume that categorical values go from 0..k-1.
        let max: u8 = col.iter().fold(0_u8, |maxval, entry| match entry {
            Entry::SmallUInt(x) => {
                if *x > maxval {
                    *x
                } else {
                    maxval
                }
            }
            _ => maxval,
        });
        Ok((max as usize + 1, None))
    } else if tally.n == tally.n_small_uint + tally.n_empty + tally.n_other {
        let mut unique_values = {
            let mut values = col.unique_values();
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            values
        };
        let mut value_map: BTreeMap<usize, String> = BTreeMap::new();
        let mut id: u8 = 0; // keep this as u8 to detect overflow
        for value in unique_values.drain(..) {
            match value {
                Entry::Other(x) => {
                    value_map.insert(id as usize, x);
                    id = id.checked_add(1).ok_or_else(|| {
                        FromCsvError::CategoricalOverflow {
                            col_name: col_name.to_owned(),
                        }
                    })?;
                }
                Entry::SmallUInt(x) => {
                    value_map.insert(id as usize, format!("{}", x));
                    id = id.checked_add(1).ok_or_else(|| {
                        FromCsvError::CategoricalOverflow {
                            col_name: col_name.to_owned(),
                        }
                    })?;
                }
                Entry::EmptyCell => (),
                _ => panic!("Tried to convert unsupported type to categorical"),
            };
        }
        Ok((value_map.len(), Some(value_map)))
    } else {
        eprintln!(
            "Not sure how to parse a column with the cell types: {:?}",
            tally
        );
        Err(FromCsvError::UnableToInferColumnType {
            col_name: col_name.to_owned(),
        })
    }
    .map(|(k, value_map)| ColType::Categorical {
        k,
        value_map,
        prior: None,
        // Note: using the vague prior causes issues with Inf propsal score when doing
        // Metroplis-Hastings because of the behavior toward zero.
        hyper: Some(CsdHyper::new(1.0, 1.0)),
    })
}

macro_rules! build_simple_coltype {
    ($parsed_col: ident, $hyper_type: ty, $xtype: ty, $col_variant: ident, $name: expr) => {{
        let mut parsed_col = $parsed_col;
        let xs: Vec<$xtype> = parsed_col
            .drain(..)
            .filter_map(|entry| {
                let x = entry.clone();
                match entry.try_into() {
                    Ok(val) => Some(val),
                    Err(EntryConversionError::EmptyCell) => None,
                    _ => {
                        panic!("invalid Entry conversion {:?} => {}", x, $name)
                    }
                }
            })
            .collect();
        let hyper = <$hyper_type>::from_data(&xs);
        Ok(ColType::$col_variant {
            hyper: Some(hyper),
            prior: None,
        })
    }};
}

fn entries_to_coltype(
    name: &str,
    col: Vec<Entry>,
    tally: EntryTally,
    cat_cutoff: usize,
    do_sanity_checks: bool,
) -> Result<ColType, FromCsvError> {
    // Run heuristics to detect potential issues with data
    if do_sanity_checks {
        heuristic_sanity_checks(name, &tally, &col);
    }

    let col_type = tally.column_type(&col, cat_cutoff);
    match col_type {
        ColumnType::Categorical => {
            column_to_categorical_coltype(col, &tally, name)
        }
        ColumnType::Continuous => {
            build_simple_coltype!(col, NixHyper, f64, Continuous, "continuous")
        }
        ColumnType::Count => {
            build_simple_coltype!(col, PgHyper, u32, Count, "count")
        }
        ColumnType::Unknown if tally.n_other > 255 => {
            Err(FromCsvError::CategoricalOverflow {
                col_name: name.to_owned(),
            })
        }
        ColumnType::Unknown => Err(FromCsvError::UnableToInferColumnType {
            col_name: name.to_owned(),
        }),
        ColumnType::Blank => Err(FromCsvError::BlankColumn {
            col_name: name.to_owned(),
        }),
    }
}

enum DataSourceReader {
    Csv(File),
    GzipCsv(GzDecoder<File>),
    Cursor(Cursor<Vec<u8>>),
}

impl Read for DataSourceReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            DataSourceReader::Csv(r) => r.read(buf),
            DataSourceReader::GzipCsv(r) => r.read(buf),
            DataSourceReader::Cursor(r) => r.read(buf),
        }
    }
}

pub enum ReaderGenerator {
    Csv(PathBuf),
    GzipCsv(PathBuf),
    Cursor(String),
}

impl From<PathBuf> for ReaderGenerator {
    fn from(path: PathBuf) -> Self {
        if path.extension().map_or(false, |ext| ext == "gz") {
            ReaderGenerator::GzipCsv(path)
        } else {
            ReaderGenerator::Csv(path)
        }
    }
}

impl ReaderGenerator {
    fn generate_csv_reader(
        &self,
    ) -> Result<Reader<DataSourceReader>, FromCsvError> {
        let inner_reader = match &self {
            ReaderGenerator::Csv(path) => File::open(path)
                .map_err(FromCsvError::Io)
                .map(|r| DataSourceReader::Csv(r)),
            ReaderGenerator::GzipCsv(path) => File::open(path)
                .map_err(FromCsvError::Io)
                .map(|r| DataSourceReader::GzipCsv(GzDecoder::new(r))),
            ReaderGenerator::Cursor(s) => Ok(DataSourceReader::Cursor(
                Cursor::new(s.clone().into_bytes()),
            )),
        }?;

        Ok(ReaderBuilder::new()
            .has_headers(true)
            .from_reader(inner_reader))
    }
}

/// Generates a default codebook from a csv file.
pub fn codebook_from_csv(
    // mut reader: Reader<R>,
    reader_generator: ReaderGenerator,
    cat_cutoff: Option<u8>,
    alpha_prior_opt: Option<CrpPrior>,
    do_sanity_checks: bool,
) -> Result<Codebook, FromCsvError> {
    let mut reader = reader_generator.generate_csv_reader()?;

    // Collect the column names
    let col_names: Vec<String> = reader
        .headers()
        .unwrap()
        .clone()
        .iter()
        .skip(1) // skip index column
        .map(String::from)
        .collect();

    let n_rows = reader.records().count();
    let mut reader = reader_generator.generate_csv_reader()?;

    let mut row_names = RowNameList::with_capacity(n_rows);
    let mut col_entries: Vec<Vec<Entry>> =
        vec![Vec::with_capacity(n_rows); col_names.len()];
    let mut tallies: Vec<EntryTally> = vec![EntryTally::new(); col_names.len()];

    let cutoff = cat_cutoff.unwrap_or(20) as usize;

    let mut col_metadata = ColMetadataList::default();

    // TODO: Could get more performance by parallelizing this loop
    reader.records().try_for_each(|rec| {
        rec.map_err(FromCsvError::CsvError)
            .and_then(|record| {
                let row_name = String::from(record.get(0).unwrap());

                row_names
                    .insert(row_name)
                    .map_err(|err| FromCsvError::DuplicatRow {
                        row_name: err.0,
                    })
                    .map(|_| record)
            })
            .map(|record| {
                record.iter().skip(1).enumerate().for_each(|(ix, cell)| {
                    let entry = Entry::from(cell);
                    tallies[ix].incr(&entry);
                    col_entries[ix].push(entry);
                })
            })
    })?;

    // collect into a vec so we can process in parallel and then push to the
    // metadata list in the proper order
    let mut colmds: Vec<(usize, ColMetadata)> = col_names
        .par_iter()
        .zip(col_entries.par_drain(..))
        .zip(tallies.par_drain(..))
        .enumerate()
        .map(|(ix, ((name, col), tally))| {
            entries_to_coltype(name, col, tally, cutoff, do_sanity_checks).map(
                |coltype| {
                    let md = ColMetadata {
                        name: name.clone(),
                        coltype,
                        notes: None,
                    };

                    (ix, md)
                },
            )
        })
        .collect::<Result<_, _>>()?;

    colmds.sort_by_key(|x| x.0);
    colmds.drain(..).try_for_each(|(_, md)| {
        col_metadata
            .push(md)
            .map_err(|col_name| FromCsvError::DuplicatColumn { col_name })
    })?;

    let alpha_prior = alpha_prior_opt
        .unwrap_or_else(|| braid_consts::geweke_alpha_prior().into());

    Ok(Codebook {
        table_name: String::from("my_data"),
        view_alpha_prior: Some(alpha_prior.clone()),
        state_alpha_prior: Some(alpha_prior),
        col_metadata,
        comments: Some(String::from("Auto-generated codebook")),
        row_names,
    })
}

// Sanity Checks on data
fn heuristic_sanity_checks(name: &str, tally: &EntryTally, column: &[Entry]) {
    // 90% of each column is non-empty
    let ratio_missing = (tally.n_empty as f64) / (tally.n as f64);
    if ratio_missing > 0.50 {
        eprintln!(
            "NOTE: Column \"{}\" is missing {:4.1}% of its values",
            name,
            100.0 * ratio_missing
        );
    }

    // Check the number of unique values
    let mut multiple_distinct_values = false;
    let mut distinct_value = None;
    for val in column {
        match val {
            Entry::EmptyCell => {}
            _ => {
                let mut hasher = DefaultHasher::new();
                val.hash(&mut hasher);
                let h = hasher.finish();
                match distinct_value {
                    Some(x) if x != h => {
                        multiple_distinct_values = true;
                        break;
                    }
                    None => {
                        distinct_value = Some(h);
                    }
                    _ => {}
                }
            }
        }
    }

    if !multiple_distinct_values {
        eprintln!("WARNING: Column \"{}\" only takes on one value...", name);
    }
}

#[cfg(test)]
mod tests {
    extern crate maplit;

    use super::*;

    use csv::ReaderBuilder;

    fn entry_tally(entries: &[Entry]) -> EntryTally {
        let mut tally = EntryTally::new();
        for entry in entries.iter() {
            tally.incr(entry);
        }

        tally
    }

    fn cell_tally(cells: &[String]) -> (Vec<Entry>, EntryTally) {
        let entries: Vec<_> =
            cells.iter().map(|s| Entry::from(s.as_str())).collect();
        let tally = entry_tally(&entries);
        (entries, tally)
    }

    #[test]
    fn entry_from_string() {
        assert_eq!(Entry::from("0 "), Entry::SmallUInt(0));
        assert_eq!(Entry::from("256"), Entry::UInt(256));
        assert_eq!(Entry::from("1356"), Entry::UInt(1356));
        assert_eq!(Entry::from("2.0"), Entry::Float(2.0));
        assert_eq!(Entry::from(" 2.2 "), Entry::Float(2.2));
        assert_eq!(Entry::from("-1"), Entry::Int(-1));
        assert_eq!(Entry::from(""), Entry::EmptyCell);
        assert_eq!(Entry::from(" "), Entry::EmptyCell);
        assert_eq!(Entry::from("mouse"), Entry::Other(String::from("mouse")));
    }

    #[test]
    fn tally() {
        let entries = vec![
            Entry::SmallUInt(1),
            Entry::EmptyCell,
            Entry::SmallUInt(4),
            Entry::Float(1.2),
            Entry::EmptyCell,
            Entry::Other(String::from("tree")),
            Entry::Int(-1),
        ];

        let tally = entry_tally(&entries);

        assert_eq!(tally.n, 7);
        assert_eq!(tally.n_small_uint, 2);
        assert_eq!(tally.n_int, 1);
        assert_eq!(tally.n_float, 1);
        assert_eq!(tally.n_empty, 2);
        assert_eq!(tally.n_other, 1);
    }

    #[test]
    fn tally_int_type_count() {
        let tally = EntryTally {
            n: 0, // Ignored for this test
            n_float: 12,
            n_int: 3,
            n_uint: 4,
            n_small_uint: 7,
            n_other: 14,
            n_empty: 2,
        };
        assert_eq!(tally.n_int_type(), 14);
    }

    #[test]
    fn col_with_any_negative_int_should_be_continuous() {
        let col = vec![
            String::from("1"),
            String::from("350"),
            String::from("1"),
            String::from("1"),
            String::from("0"),
            String::from("-1"),
        ];
        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();
        assert!(coltype.is_continuous());
    }

    #[test]
    fn col_with_any_negative_int_should_be_continuous_missing() {
        let col = vec![
            String::from("1"),
            String::from("350"),
            String::from(""),
            String::from("1"),
            String::from("0"),
            String::from("-1"),
        ];
        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();
        assert!(coltype.is_continuous());
    }

    #[test]
    fn non_rounded_vec_should_be_continuous() {
        let col = vec![
            String::from("0.1"),
            String::from("1.0"),
            String::from("2.1"),
            String::from("4.2"),
        ];
        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();
        assert!(coltype.is_continuous());
    }

    #[test]
    fn non_rounded_vec_should_be_continuous_with_empty() {
        let col = vec![
            String::from("0.1"),
            String::from("1.0"),
            String::from("2.1"),
            String::from(""),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_continuous());
    }

    #[test]
    fn some_non_rounded_vec_should_be_continuous() {
        let col = vec![
            String::from("0.1"),
            String::from("1"),
            String::from("2.1"),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_continuous());
    }

    #[test]
    fn some_non_rounded_vec_should_be_continuous_with_empty() {
        let col = vec![
            String::from(""),
            String::from("1"),
            String::from("2.1"),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_continuous());
    }

    #[test]
    fn all_rounded_vec_should_be_count_if_low_cutoff() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from("2"),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype_count =
            entries_to_coltype(&"".to_owned(), entries, tally, 3, true)
                .unwrap();

        assert!(coltype_count.is_count());

        let (entries, tally) = cell_tally(&col);
        let coltype_cat =
            entries_to_coltype(&"".to_owned(), entries, tally, 5, true)
                .unwrap();

        assert!(coltype_cat.is_categorical());
    }

    #[test]
    fn all_rounded_vec_should_be_count_if_and_gt_255() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from("256"),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype_count =
            entries_to_coltype(&"".to_owned(), entries, tally, 50, true)
                .unwrap();

        assert!(coltype_count.is_count());
    }

    #[test]
    fn all_rounded_vec_should_be_count_if_low_with_empty() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from(""),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype_count =
            entries_to_coltype(&"".to_owned(), entries, tally, 2, true)
                .unwrap();

        assert!(coltype_count.is_count());

        let (entries, tally) = cell_tally(&col);
        let coltype_cat =
            entries_to_coltype(&"".to_owned(), entries, tally, 5, true)
                .unwrap();

        assert!(coltype_cat.is_categorical());
    }

    #[test]
    fn vec_with_string_should_be_categorical() {
        let col = vec![
            String::from("cat"),
            String::from("dog"),
            String::from("hamster"),
            String::from("bird"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());
    }

    #[test]
    fn vec_with_string_should_be_categorical_with_empty() {
        let col = vec![
            String::from("cat"),
            String::from("dog"),
            String::from(""),
            String::from("bird"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());
    }

    #[test]
    fn vec_with_string_should_be_categorical_with_int() {
        let col = vec![
            String::from("cat"),
            String::from("dog"),
            String::from("12"),
            String::from("bird"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());
    }

    #[test]
    fn all_int_vec_should_have_no_value_map() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from("2"),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            // Assumes vales can take on [0..max - 1]
            assert_eq!(k, 5);
            assert!(value_map.is_none());
        }
    }

    #[test]
    fn all_int_vec_should_have_no_value_map_with_empty() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from(""),
            String::from("4"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            // Assumes vales can take on [0..max - 1]
            assert_eq!(k, 5);
            assert!(value_map.is_none());
        }
    }

    #[test]
    fn vec_with_all_other_type_should_have_value_map() {
        let col = vec![
            String::from("dog"),
            String::from("cat"),
            String::from("fox"),
            String::from("bear"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 4);
            assert!(value_map.is_some());
        }
    }

    #[test]
    fn value_map_should_be_ordered() {
        let col = vec![
            String::from("dog"),
            String::from("cat"),
            String::from("fox"),
            String::from("bear"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 4);
            let vm = value_map.unwrap();
            assert_eq!(vm.get(&0).unwrap(), "bear");
            assert_eq!(vm.get(&1).unwrap(), "cat");
            assert_eq!(vm.get(&2).unwrap(), "dog");
            assert_eq!(vm.get(&3).unwrap(), "fox");
        }
    }

    #[test]
    fn vec_with_all_other_type_should_have_value_map_with_empty() {
        let col = vec![
            String::from("dog"),
            String::from(""),
            String::from("fox"),
            String::from("bear"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 3);
            assert!(value_map.is_some());
        }
    }

    #[test]
    fn vec_with_some_other_type_should_have_value_map() {
        let col = vec![
            String::from("dog"),
            String::from("2"),
            String::from("fox"),
            String::from("bear"),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 4);
            assert!(value_map.is_some());
        }
    }

    #[test]
    fn vec_with_some_other_type_should_have_value_map_with_empty() {
        let col = vec![
            String::from("dog"),
            String::from("2"),
            String::from(""),
            String::from(""),
        ];

        let (entries, tally) = cell_tally(&col);
        let coltype =
            entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
                .unwrap();

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 2);
            assert!(value_map.is_some());
        }
    }

    #[test]
    fn all_empty_column_error() {
        let col = vec![
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
        ];

        let (entries, tally) = cell_tally(&col);
        let err = entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
            .unwrap_err();

        match err {
            FromCsvError::BlankColumn { .. } => (),
            _ => panic!("Wrong error: '{}'", err),
        }
    }

    #[test]
    fn string_column_with_too_many_unique_values_is_unknown() {
        // 255 unique values is ok
        {
            let col: Vec<_> = (0..255).map(|i| format!("s_{}", i)).collect();

            let (entries, tally) = cell_tally(&col);
            let res =
                entries_to_coltype(&"".to_owned(), entries, tally, 10, true);

            match res {
                Ok(ColType::Categorical { .. }) => (),
                Ok(_) => panic!("wrong column type"),
                Err(err) => panic!("error: {}", err),
            }
        }

        // 256 unique values is not ok
        {
            let col: Vec<_> = (0..=255).map(|i| format!("s_{}", i)).collect();

            let (entries, tally) = cell_tally(&col);
            let res =
                entries_to_coltype(&"".to_owned(), entries, tally, 10, true);

            match res {
                Err(FromCsvError::CategoricalOverflow { .. }) => (),
                Ok(_) => panic!("should have errored"),
                Err(err) => panic!("wrong error: {}", err),
            }
        }
    }

    #[test]
    #[should_panic]
    fn empty_column_panics() {
        let col = Vec::new();

        let (entries, tally) = cell_tally(&col);
        let _res = entries_to_coltype(&"".to_owned(), entries, tally, 10, true)
            .unwrap();
    }

    const CSV_DATA: &str = "\
        id,x,y
        0,1.1,cat
        1,2.2,dog
        2,3.4,
        3,0.1,cat
        4,,dog
        5,,dog
        6,0.3,dog
        7,-1.2,dog
        8,1.0,dog
        9,,human\
    ";

    use std::io::Cursor;

    // make sure that the value map indices line up correctly even if there
    // are missing values
    #[test]
    fn default_codebook_string_csv_valuemap_indices() {
        let csv_data = String::from(CSV_DATA);
        let csv_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(csv_data.as_bytes()));

        let codebook = codebook_from_csv(csv_reader, None, None, true).unwrap();
        let colmds = codebook.col_metadata(String::from("y")).unwrap();
        if let ColType::Categorical {
            value_map: Some(vm),
            ..
        } = &colmds.coltype
        {
            assert_eq!(vm.len(), 3);
            assert!(vm.contains_key(&0_usize));
            assert!(vm.contains_key(&1_usize));
            assert!(vm.contains_key(&2_usize));
        } else {
            assert!(false);
        }
    }

    #[test]
    fn codebook_with_all_types_inferse_correct_types() {
        let data = "\
            id,cat_str,cat_int,count,cts_int,cts_float
            0,       A,      1,    0,    -1,      1.0
            1,        ,      0,  256,     0,      2.0
            2,       B,      1,    2,    12,      3.0
            3,       A,      1,     ,      ,
            4,       A,       ,   43,     3,\
        ";

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(data.as_bytes()));

        let codebook = codebook_from_csv(reader, None, None, true).unwrap();

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
    fn codebook_from_csv_with_bad_csv_returns_error() {
        // missing a column in the first row
        let data = "\
            id,x,y
            0,1
            1,,0
            2,,
            3,,1
            4,,1\
        ";

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(data.as_bytes()));

        match codebook_from_csv(reader, None, None, true) {
            Err(FromCsvError::CsvError(_)) => (),
            _ => panic!("should have detected bad input"),
        }
    }

    #[test]
    fn codebook_from_csv_with_blank_column_returns_error() {
        let data = "\
            id,x,y
            0,,1
            1,,0
            2,,
            3,,1
            4,,1\
        ";

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(data.as_bytes()));

        match codebook_from_csv(reader, None, None, true) {
            Err(FromCsvError::BlankColumn { col_name }) => {
                assert_eq!(col_name, String::from("x"))
            }
            _ => panic!("should have detected blank column"),
        }
    }

    #[test]
    fn codebook_from_csv_with_too_many_cat_values_returns_error() {
        let mut data = String::from("ix,x,y\n");
        for i in 0..=256 {
            let catval = format!("{}val", i);
            let row = format!("{},{},{}", i, catval, i % 4);

            data.push_str(&row);

            if i < 256 {
                data.push_str("\n");
            }
        }

        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(Cursor::new(data.as_bytes()));

        match codebook_from_csv(reader, None, None, true) {
            Err(FromCsvError::CategoricalOverflow { col_name }) => {
                assert_eq!(col_name, String::from("x"))
            }
            Err(_) => panic!("wrong error"),
            _ => panic!("should have detected categorical overflow"),
        }
    }
}

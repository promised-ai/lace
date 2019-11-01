use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::convert::{From, TryInto};
use std::f64;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::mem::transmute_copy;
use std::str::FromStr;

use braid_stats::labeler::Label;
use braid_stats::prior::{CrpPrior, NigHyper};
use braid_utils::unique::UniqueCollection;
use csv::Reader;

use crate::codebook::{Codebook, ColMetadata, ColType, SpecType};
use crate::gmd::process_gmd_csv;

// The type of entry in the CSV cell. Currently Int only supports u8 because
// `categorical` is the only integer type.
#[derive(Clone, Debug, PartialOrd)]
enum Entry {
    Float(f64),
    Int(u8),
    Label(Label),
    Other(String),
    EmptyCell,
}

impl Eq for Entry {}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Entry::Float(x) => {
                if let Entry::Float(y) = other {
                    unsafe {
                        let xc: u64 = transmute_copy(x);
                        let yc: u64 = transmute_copy(y);
                        xc == yc
                    }
                } else {
                    false
                }
            }
            Entry::Int(x) => {
                if let Entry::Int(y) = other {
                    x == y
                } else {
                    false
                }
            }
            Entry::Label(x) => {
                if let Entry::Label(y) = other {
                    x == y
                } else {
                    false
                }
            }
            Entry::Other(x) => {
                if let Entry::Other(y) = other {
                    x == y
                } else {
                    false
                }
            }
            Entry::EmptyCell => match other {
                Entry::EmptyCell => true,
                _ => false,
            },
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
            Entry::Label(x) => x.hash(state),
            Entry::Other(x) => x.hash(state),
            Entry::EmptyCell => 0_u8.hash(state),
        }
    }
}

impl From<String> for Entry {
    fn from(val: String) -> Entry {
        let s = val.trim();
        if s == "" {
            return Entry::EmptyCell;
        }

        // preference: int -> float -> Label -> other
        if let Ok(x) = u8::from_str(s) {
            return Entry::Int(x);
        }

        if let Ok(x) = f64::from_str(s) {
            return Entry::Float(x);
        }

        if let Ok(x) = Label::from_str(s) {
            return Entry::Label(x);
        }

        Entry::Other(s.to_owned())
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
            Entry::EmptyCell => Err(EntryConversionError::EmptyCell),
            _ => Err(EntryConversionError::InvalidInnerType),
        }
    }
}

impl_try_into_entry!(u8, Int);
impl_try_into_entry!(Label, Label);
impl_try_into_entry!(String, Other);

fn parse_column(mut col: Vec<String>) -> Vec<Entry> {
    col.drain(..).map(Entry::from).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ColumnType {
    Categorical,
    Continuous,
    Labeler,
    Unknown,
    Blank,
}

#[derive(Debug)]
struct EntryTally {
    pub n: usize,
    pub n_float: usize,
    pub n_int: usize,
    pub n_label: usize,
    pub n_other: usize,
    pub n_empty: usize,
}

impl EntryTally {
    pub fn new(n: usize) -> EntryTally {
        EntryTally {
            n,
            n_float: 0,
            n_int: 0,
            n_label: 0,
            n_other: 0,
            n_empty: 0,
        }
    }

    pub fn incr(&mut self, entry: &Entry) {
        match entry {
            Entry::Float(..) => self.n_float += 1,
            Entry::Int(..) => self.n_int += 1,
            Entry::Label(..) => self.n_label += 1,
            Entry::Other(..) => self.n_other += 1,
            Entry::EmptyCell => self.n_empty += 1,
        }
    }

    pub fn tally(mut self, col: &[Entry]) -> Self {
        col.iter().for_each(|entry| self.incr(entry));
        self
    }

    pub fn column_type(&self, col: &[Entry], cat_cutoff: usize) -> ColumnType {
        // FIXME: This is stupid complicated
        if self.n_label > 0 {
            if self.n_label + self.n_empty != self.n {
                ColumnType::Unknown
            } else {
                ColumnType::Labeler
            }
        } else if self.n_empty == self.n {
            ColumnType::Blank
        } else if self.n_int + self.n_empty == self.n {
            let n_unique = col.n_unique_cutoff(cat_cutoff);
            if n_unique < cat_cutoff {
                ColumnType::Categorical
            } else {
                ColumnType::Continuous
            }
        } else if self.n_float + self.n_int + self.n_empty == self.n {
            ColumnType::Continuous
        } else if self.n_other > 0 {
            let n_unique = col.n_unique_cutoff(cat_cutoff);
            if n_unique <= cat_cutoff {
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
) -> ColType {
    use braid_stats::prior::CsdHyper;

    let (k, value_map) = if tally.n == tally.n_int + tally.n_empty {
        // Assume that categorical values go from 0..k-1.
        let max: u8 = col.iter().fold(0_u8, |maxval, entry| match entry {
            Entry::Int(x) => {
                if *x > maxval {
                    *x
                } else {
                    maxval
                }
            }
            _ => maxval,
        });
        (max as usize + 1, None)
    } else if tally.n == tally.n_int + tally.n_empty + tally.n_other {
        let mut unique_values = col.unique_values();
        let mut value_map: BTreeMap<usize, String> = BTreeMap::new();
        let mut id: u8 = 0; // keep this as u8 to detect overflow
        for value in unique_values.drain(..) {
            match value {
                Entry::Other(x) => {
                    value_map.insert(id as usize, x);
                    id = id.checked_add(1).expect("too man categorical values");
                }
                Entry::Int(x) => {
                    value_map.insert(id as usize, format!("{}", x));
                    id = id.checked_add(1).expect("too man categorical values");
                }
                Entry::EmptyCell => (),
                _ => panic!("Cannot create value map from unhashable type"),
            };
        }
        (value_map.len(), Some(value_map))
    } else {
        panic!(
            "Not sure how to parse a column with the cell types: {:?}",
            tally
        );
    };

    ColType::Categorical {
        k,
        value_map,
        hyper: Some(CsdHyper::vague(k)),
    }
}

fn column_to_labeler_coltype(parsed_col: Vec<Entry>) -> ColType {
    let n_labels: u8 = parsed_col.iter().fold(0, |max, entry| match entry {
        Entry::Label(label) => {
            let truth = label.truth.unwrap_or(0);
            max.max(label.label.max(truth))
        }
        Entry::EmptyCell => max,
        _ => panic!("Invalid entry: {:?}", entry),
    }) + 1;
    ColType::Labeler {
        n_labels,
        pr_h: None,
        pr_k: None,
        pr_world: None,
    }
}

fn entries_to_coltype(
    name: &str,
    col: Vec<String>,
    cat_cutoff: usize,
) -> ColType {
    let parsed_col = parse_column(col);

    let tally = EntryTally::new(parsed_col.len()).tally(&parsed_col);

    // Run heuristics to detect potential issues with data
    heuristic_sanity_checks(name, &tally, &parsed_col);

    match tally.column_type(&parsed_col, cat_cutoff) {
        ColumnType::Categorical => {
            column_to_categorical_coltype(parsed_col, &tally)
        }
        ColumnType::Continuous => {
            let mut parsed_col = parsed_col;
            let xs: Vec<f64> = parsed_col
                .drain(..)
                .filter_map(|entry| match entry.try_into() {
                    Ok(val) => Some(val),
                    Err(EntryConversionError::EmptyCell) => None,
                    _ => panic!("invalid Entry -> f64 conversion"),
                })
                .collect();
            let hyper = NigHyper::from_data(&xs);
            ColType::Continuous { hyper: Some(hyper) }
        }
        ColumnType::Labeler => column_to_labeler_coltype(parsed_col),
        ColumnType::Unknown => panic!("Could not figure out column type"),
        ColumnType::Blank => panic!("Blank column"),
    }
}

struct TransposedCsv {
    pub col_names: Vec<String>,
    pub row_names: Vec<String>,
    pub data: Vec<Vec<String>>,
}

// Assumes `mat` is square
fn transpose<T>(mut mat: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let ncols = mat[0].len();
    (0..ncols)
        .map(|_| mat.iter_mut().map(|row| row.remove(0)).collect())
        .collect()
}

fn transpose_csv<R: Read>(mut reader: Reader<R>) -> TransposedCsv {
    let mut row_names: Vec<String> = Vec::new();
    let mut data: Vec<Vec<String>> = Vec::new();

    reader.records().for_each(|rec| {
        let record = rec.unwrap();
        let row_name: String = String::from(record.get(0).unwrap());

        row_names.push(row_name);

        let row_data: Vec<String> =
            record.iter().skip(1).map(String::from).collect();

        data.push(row_data)
    });

    let col_names: Vec<String> = reader
        .headers()
        .unwrap()
        .to_owned()
        .iter()
        .skip(1)
        .map(String::from)
        .collect();

    TransposedCsv {
        col_names,
        row_names,
        data: transpose(data),
    }
}

/// Generates a default codebook from a csv file.
pub fn codebook_from_csv<R: Read>(
    reader: Reader<R>,
    cat_cutoff: Option<u8>,
    alpha_prior_opt: Option<CrpPrior>,
    gmd_reader: Option<Reader<R>>,
) -> Codebook {
    let gmd = match gmd_reader {
        Some(r) => process_gmd_csv(r),
        None => BTreeMap::new(),
    };

    let mut csv_t = transpose_csv(reader);

    let cutoff = cat_cutoff.unwrap_or(20) as usize;

    let mut col_metadata: BTreeMap<String, ColMetadata> = BTreeMap::new();

    csv_t
        .col_names
        .drain(..)
        .zip(csv_t.data.drain(..))
        .enumerate()
        .for_each(|(id, (name, col))| {
            let coltype = entries_to_coltype(&name, col, cutoff);

            let spec_type = if coltype.is_categorical() {
                match gmd.get(&name) {
                    Some(gmd_row) => SpecType::Genotype {
                        chrom: gmd_row.chrom,
                        pos: gmd_row.pos,
                    },
                    None => SpecType::Other,
                }
            } else if coltype.is_continuous() {
                SpecType::Phenotype
            } else {
                SpecType::Other
            };

            let md = ColMetadata {
                id,
                spec_type,
                name: name.clone(),
                coltype,
                notes: None,
            };

            col_metadata.insert(name, md);
        });

    let alpha_prior = alpha_prior_opt
        .unwrap_or_else(|| braid_consts::geweke_alpha_prior().into());

    Codebook {
        table_name: String::from("my_data"),
        view_alpha_prior: Some(alpha_prior.clone()),
        state_alpha_prior: Some(alpha_prior),
        col_metadata,
        comments: Some(String::from("Auto-generated codebook")),
        row_names: Some(csv_t.row_names),
    }
}

// Sanity Checks on data
fn heuristic_sanity_checks(name: &str, tally: &EntryTally, column: &[Entry]) {
    // 90% of each column is non-empty
    let ratio_missing = (tally.n_empty as f64) / (tally.n as f64);
    if ratio_missing > 0.1 {
        eprintln!("WARNING: Column \"{}\" is missing {:4.1}% of its values, this might be an error...", name, 100.0 * ratio_missing);
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

    use std::path::Path;

    use crate::codebook::SpecType;
    use csv::ReaderBuilder;

    #[test]
    fn entry_from_string() {
        assert_eq!(Entry::from(String::from("0 ")), Entry::Int(0));
        assert_eq!(Entry::from(String::from("0")), Entry::Int(0));
        assert_eq!(Entry::from(String::from("1")), Entry::Int(1));
        assert_eq!(Entry::from(String::from("2.0")), Entry::Float(2.0));
        assert_eq!(Entry::from(String::from(" 2.2 ")), Entry::Float(2.2));
        assert_eq!(Entry::from(String::from("-1")), Entry::Float(-1.0));
        assert_eq!(Entry::from(String::from("")), Entry::EmptyCell);
        assert_eq!(Entry::from(String::from(" ")), Entry::EmptyCell);
        assert_eq!(
            Entry::from(String::from("mouse")),
            Entry::Other(String::from("mouse"))
        );
    }

    #[test]
    fn tally() {
        let entries = vec![
            Entry::Int(1),
            Entry::EmptyCell,
            Entry::Int(4),
            Entry::Float(1.2),
            Entry::EmptyCell,
            Entry::Other(String::from("tree")),
            Entry::Int(0),
        ];

        let tally = EntryTally::new(entries.len()).tally(&entries);

        assert_eq!(tally.n, 7);
        assert_eq!(tally.n_int, 3);
        assert_eq!(tally.n_float, 1);
        assert_eq!(tally.n_empty, 2);
        assert_eq!(tally.n_other, 1);
    }

    #[test]
    fn non_rounded_vec_should_be_continuous() {
        let col = vec![
            String::from("0.1"),
            String::from("1.0"),
            String::from("2.1"),
            String::from("4.2"),
        ];
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
        assert!(coltype.is_continuous());
    }

    #[test]
    fn all_rounded_vec_should_be_continuous_if_low_cutoff() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from("2"),
            String::from("4"),
        ];
        let coltype_cont = entries_to_coltype(&"".to_owned(), col.clone(), 3);
        assert!(coltype_cont.is_continuous());

        let coltype_cat = entries_to_coltype(&"".to_owned(), col, 5);
        assert!(coltype_cat.is_categorical());
    }

    #[test]
    fn all_rounded_vec_should_be_continuous_if_low_with_empty() {
        let col = vec![
            String::from("0"),
            String::from("1"),
            String::from(""),
            String::from("4"),
        ];
        let coltype_cont = entries_to_coltype(&"".to_owned(), col.clone(), 2);
        assert!(coltype_cont.is_continuous());

        let coltype_cat = entries_to_coltype(&"".to_owned(), col, 5);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);

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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);

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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 4);
            assert!(value_map.is_some());
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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);

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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);

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
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);

        assert!(coltype.is_categorical());

        if let ColType::Categorical { k, value_map, .. } = coltype {
            assert_eq!(k, 2);
            assert!(value_map.is_some());
        }
    }

    #[test]
    #[should_panic]
    fn all_empty_column_panics() {
        let col = vec![
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
        ];
        let _coltype = entries_to_coltype(&"".to_owned(), col, 10);
    }

    #[test]
    fn all_label_column_should_be_labeler_type() {
        let col = vec![
            String::from("IL(1, None)"),
            String::from("IL(0, 0)"),
            String::from("IL(1, 1)"),
            String::from("IL(0, None)"),
        ];
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
        assert!(coltype.is_labeler());
        if let ColType::Labeler { n_labels, .. } = coltype {
            assert_eq!(n_labels, 2);
        }
    }

    #[test]
    fn correct_number_of_labeler_type_high_in_truth() {
        let col = vec![
            String::from("IL(0, None)"),
            String::from("IL(0, 0)"),
            String::from("IL(1, 3)"),
            String::from("IL(0, None)"),
        ];
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
        assert!(coltype.is_labeler());
        if let ColType::Labeler { n_labels, .. } = coltype {
            assert_eq!(n_labels, 4);
        }
    }

    #[test]
    fn correct_number_of_labeler_type_high_in_label() {
        let col = vec![
            String::from("IL(0, None)"),
            String::from("IL(0, 0)"),
            String::from("IL(1, 2)"),
            String::from("IL(5, None)"),
        ];
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
        assert!(coltype.is_labeler());
        if let ColType::Labeler { n_labels, .. } = coltype {
            assert_eq!(n_labels, 6);
        }
    }

    #[test]
    fn all_label_column_with_missing_should_be_labeler_type() {
        let col = vec![
            String::from("IL(1, None)"),
            String::from("IL(0, 0)"),
            String::from(""),
            String::from("IL(0, None)"),
        ];
        let coltype = entries_to_coltype(&"".to_owned(), col, 10);
        assert!(coltype.is_labeler());
        if let ColType::Labeler { n_labels, .. } = coltype {
            assert_eq!(n_labels, 2);
        }
    }

    #[test]
    #[should_panic]
    fn all_label_column_with_other_should_panic() {
        let col = vec![
            String::from("IL(1, None)"),
            String::from("IL(0, 0)"),
            String::from("12"),
            String::from("IL(0, None)"),
        ];
        let _coltype = entries_to_coltype(&"".to_owned(), col, 10);
    }

    #[test]
    #[should_panic]
    fn empty_column_panics() {
        let _coltype = entries_to_coltype(&"".to_owned(), vec![], 10);
    }

    #[test]
    fn correct_codebook_with_genomic_metadata() {
        let gmd_reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(Path::new("resources/test/genomics-md.csv"))
            .unwrap();

        let csv_reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(Path::new("resources/test/genomics.csv"))
            .unwrap();

        let cb = codebook_from_csv(csv_reader, None, None, Some(gmd_reader));

        let spec_type =
            |col: &str| cb.col_metadata[&String::from(col)].spec_type.clone();

        assert_eq!(
            spec_type("m_0"),
            SpecType::Genotype {
                pos: 0.12,
                chrom: 1
            }
        );
        assert_eq!(
            spec_type("m_1"),
            SpecType::Genotype {
                pos: 0.23,
                chrom: 1
            }
        );
        assert_eq!(
            spec_type("m_2"),
            SpecType::Genotype {
                pos: 0.45,
                chrom: 2
            }
        );
        assert_eq!(
            spec_type("m_3"),
            SpecType::Genotype {
                pos: 0.67,
                chrom: 2
            }
        );
        assert_eq!(
            spec_type("m_4"),
            SpecType::Genotype {
                pos: 0.89,
                chrom: 3
            }
        );
        assert_eq!(
            spec_type("m_5"),
            SpecType::Genotype {
                pos: 1.01,
                chrom: 3
            }
        );
        assert_eq!(
            spec_type("m_6"),
            SpecType::Genotype {
                pos: 1.12,
                chrom: 3
            }
        );
        assert_eq!(
            spec_type("m_7"),
            SpecType::Genotype {
                pos: 1.23,
                chrom: 4
            }
        );
        assert_eq!(spec_type("other"), SpecType::Other);
        assert_eq!(spec_type("t_1"), SpecType::Phenotype);
        assert_eq!(spec_type("t_2"), SpecType::Phenotype);

        assert!(cb.col_metadata[&String::from("label")].coltype.is_labeler());
        assert_eq!(spec_type("label"), SpecType::Other);
    }

    const CSV_DATA: &str = r#"id,x,y
0,1.1,cat
1,2.2,dog
2,3.4,
3,0.1,cat
4,,dog
5,,dog
6,0.3,dog
7,-1.2,dog
8,1.0,dog
9,,human"#;

    // make sure that the value map indices line up correctly even if there
    // are missing values
    #[test]
    fn default_codebook_string_csv_valuemap_indices() {
        let csv_data = String::from(CSV_DATA);
        let csv_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let codebook = codebook_from_csv(csv_reader, None, None, None);
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
}

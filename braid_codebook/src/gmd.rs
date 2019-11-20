use std::collections::BTreeMap;
use std::io::Read;

use braid_utils::parse_result;
use csv::Reader;

/// Contains the chromosome id and position in cM of a marker.
#[derive(Debug, PartialEq)]
pub struct GmdRow {
    /// The chromosome
    pub chrom: u8,
    /// The position in cM
    pub pos: f64,
}

pub fn process_gmd_csv<R: Read>(
    mut reader: Reader<R>,
) -> BTreeMap<String, GmdRow> {
    let csv_header = reader.headers().unwrap().clone();
    let mut colixs: BTreeMap<&str, usize> = BTreeMap::new();
    csv_header.iter().enumerate().for_each(|(ix, name)| {
        colixs.insert(name, ix);
    });

    let mut gmd: BTreeMap<String, GmdRow> = BTreeMap::new();

    let chrom_ix = colixs["chrom"];
    let pos_ix = colixs["pos"];

    reader.records().for_each(|rec| {
        let rec_uw = rec.unwrap();
        let id = String::from(rec_uw.get(0).unwrap());
        let chrom_str = rec_uw.get(chrom_ix).unwrap();
        let pos_str = rec_uw.get(pos_ix).unwrap();
        // FIXME-RESULT: use result correctly
        let chrom = parse_result::<u8>(chrom_str).unwrap().unwrap();
        // FIXME-RESULT: use result correctly
        let pos = parse_result::<f64>(pos_str).unwrap().unwrap();
        gmd.insert(id, GmdRow { chrom, pos });
    });

    gmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use csv::ReaderBuilder;
    use std::path::Path;

    #[test]
    fn generate_correct_genomic_metadata() {
        let reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(Path::new("resources/test/genomics-md.csv"))
            .unwrap();
        let gmd = process_gmd_csv(reader);

        let mut gmd_t: BTreeMap<String, GmdRow> = BTreeMap::new();

        gmd_t.insert(
            String::from("m_0"),
            GmdRow {
                pos: 0.12,
                chrom: 1,
            },
        );
        gmd_t.insert(
            String::from("m_1"),
            GmdRow {
                pos: 0.23,
                chrom: 1,
            },
        );
        gmd_t.insert(
            String::from("m_2"),
            GmdRow {
                pos: 0.45,
                chrom: 2,
            },
        );
        gmd_t.insert(
            String::from("m_3"),
            GmdRow {
                pos: 0.67,
                chrom: 2,
            },
        );
        gmd_t.insert(
            String::from("m_4"),
            GmdRow {
                pos: 0.89,
                chrom: 3,
            },
        );
        gmd_t.insert(
            String::from("m_5"),
            GmdRow {
                pos: 1.01,
                chrom: 3,
            },
        );
        gmd_t.insert(
            String::from("m_6"),
            GmdRow {
                pos: 1.12,
                chrom: 3,
            },
        );
        gmd_t.insert(
            String::from("m_7"),
            GmdRow {
                pos: 1.23,
                chrom: 4,
            },
        );

        assert_eq!(gmd, gmd_t);
    }
}

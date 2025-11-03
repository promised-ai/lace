// This test reproduces behavior whereby inserting data into an engine without
// running an Engine update leaves the metadata in an invalid state.
use lace::{Given, Row};
use lace_codebook::{ColMetadata, ColMetadataList};
use lace_metadata::SerializedType;
use lace_stats::rand;
use lace_stats::rv::traits::Sampleable;

use std::convert::TryInto;

fn empty_engine() -> lace::Engine {
    use lace::data::DataSource;
    use lace_codebook::Codebook;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    lace::Engine::new(
        16,
        Codebook::default(),
        DataSource::Empty,
        0,
        Xoshiro256Plus::from_os_rng(),
    )
    .unwrap()
}

fn gen_row<R: rand::Rng>(ix: u32, mut rng: &mut R) -> Row<String, String> {
    use lace_data::Datum;
    use lace_stats::rv::dist::Gaussian;

    let g = Gaussian::default();
    let mut values = g
        .sample_stream(&mut rng)
        .enumerate()
        .take(14)
        .map(|(ix, x)| (ix.to_string(), Datum::Continuous(x)))
        .collect::<Vec<(String, Datum)>>();

    let label = (ix % 3) as u32;

    values.push((String::from("label"), Datum::Categorical(label.into())));

    (ix.to_string(), values).try_into().unwrap()
}

fn gen_col_metadata(col_name: &str) -> ColMetadata {
    use lace_codebook::ColType;
    use lace_stats::prior::csd::CsdHyper;
    use lace_stats::prior::nix::NixHyper;

    if col_name != "label" {
        // temporal variables
        ColMetadata {
            name: String::from(col_name),
            coltype: ColType::Continuous {
                hyper: Some(NixHyper::default()),
                prior: None,
            },
            notes: None,
            missing_not_at_random: false,
        }
    } else {
        // label/action
        ColMetadata {
            name: String::from("label"),
            coltype: ColType::Categorical {
                k: 5,
                hyper: Some(CsdHyper::new(2.0, 3.0)),
                prior: None,
                value_map: lace_codebook::ValueMap::UInt(5),
            },
            notes: None,
            missing_not_at_random: false,
        }
    }
}

fn gen_new_metadata<R: lace::RowIndex>(
    row: &Row<R, String>,
) -> Option<ColMetadataList> {
    let colmds: Vec<ColMetadata> = row
        .values
        .iter()
        .map(|value| gen_col_metadata(value.col_ix.as_str()))
        .collect();
    Some(colmds.try_into().unwrap())
}

#[test]
fn otacon_on_empty_table() {
    use lace::{HasData, OracleT, WriteMode};

    let mut rng = rand::rng();
    let mut engine = empty_engine();

    let n_iters = 100;

    {
        let row = gen_row(0, &mut rng);
        let new_md = gen_new_metadata(&row);
        engine
            .insert_data(vec![row], new_md, WriteMode::unrestricted())
            .unwrap();
        engine.run(1).unwrap();
    }

    let mut sum = 0.0;
    for i in 1..n_iters {
        let row = gen_row(i, &mut rng);
        engine
            .insert_data(vec![row], None, WriteMode::unrestricted())
            .unwrap();
        for ix in 0..15 {
            let vals = vec![vec![engine.cell(i as usize, ix)]];
            let logps = engine
                .logp_scaled(&[ix], &vals, &Given::<usize>::Nothing, None)
                .unwrap();
            sum += logps[0];
        }
    }

    println!("{}", sum);
}

#[test]
fn otacon_insert_after_save_load() {
    use lace::{AppendStrategy, WriteMode};

    let mut rng = rand::rng();
    let mut engine = empty_engine();

    let n_iters = 100;

    {
        let row = gen_row(0, &mut rng);
        let new_md = gen_new_metadata(&row);
        engine
            .insert_data(vec![row], new_md, WriteMode::unrestricted())
            .unwrap();
        engine.run(1).unwrap();
    }

    // generate the base model
    for i in 1..n_iters {
        let row = gen_row(i, &mut rng);
        engine
            .insert_data(vec![row], None, WriteMode::unrestricted())
            .unwrap();
    }
    engine.run(10).unwrap();

    let dir = tempfile::tempdir().unwrap();
    engine.save(dir.path(), SerializedType::Yaml).unwrap();

    engine = lace::Engine::load(dir.path()).unwrap();

    {
        let write_mode = WriteMode {
            append_strategy: AppendStrategy::Trench {
                max_n_rows: 120,
                trench_ix: 102,
            },
            ..WriteMode::unrestricted()
        };

        print!("inserting...");
        for i in 1..n_iters {
            let row = gen_row(i + n_iters, &mut rng);
            engine.insert_data(vec![row], None, write_mode).unwrap();
        }
    }
}

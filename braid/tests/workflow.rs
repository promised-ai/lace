use braid::config::EngineUpdateConfig;
use braid::data::DataSource;
use braid::Builder;
use braid::Engine;
use braid_codebook::csv::codebook_from_csv;
use braid_codebook::csv::ReaderGenerator;
use rand::SeedableRng;
use std::fs::{remove_file, File};
use std::io::Write;
use std::path::PathBuf;

const CSV_DATA: &str = r#"
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
9,,human
"#;

// Smoke test default CSV generation with string data
#[test]
fn default_csv_workflow() {
    let path = PathBuf::from("tmp.csv");
    let mut csv_file = File::create(&path).unwrap();
    let csv_data = String::from(CSV_DATA);
    csv_file.write_all(csv_data.as_bytes()).unwrap();

    let reader_generator = ReaderGenerator::Cursor(csv_data);

    // default codebook
    let codebook =
        codebook_from_csv(reader_generator, None, None, true, false).unwrap();
    let rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let mut engine =
        Engine::new(4, codebook, DataSource::Csv(path.clone()), 0, rng)
            .unwrap();
    engine.run(200).unwrap();
    remove_file(path).unwrap();
}

// Smoke test satellites dataset csv which is pretty messy and sparse. This has
// caught errors not caught by other tests.
#[test]
fn satellites_csv_workflow() {
    let path = PathBuf::from("resources/datasets/satellites/data.csv");

    let reader_generator = ReaderGenerator::Csv(path.clone());

    // default codebook
    let codebook =
        codebook_from_csv(reader_generator, None, None, true, false).unwrap();

    let mut engine: Engine = Builder::new(DataSource::Csv(path))
        .codebook(codebook)
        .with_nstates(4)
        .seed_from_u64(1776)
        .build()
        .unwrap();

    let config = EngineUpdateConfig::with_default_transitions()
        .n_iters(100)
        .timeout(Some(30));

    engine.update(config, None, None).unwrap();
}
use lace::config::EngineUpdateConfig;
use lace::data::DataSource;
use lace::update_handler::Timeout;
use lace::Engine;
use lace::EngineBuilder;
use lace_codebook::data::codebook_from_csv;
use rand::SeedableRng;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

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

fn datafile() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(CSV_DATA.as_bytes()).unwrap();
    f
}

// Smoke test default CSV generation with string data
#[test]
fn default_csv_workflow() {
    let file = datafile();

    // default codebook
    let codebook = codebook_from_csv(file.path(), None, None, false).unwrap();
    let rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let mut engine = Engine::new(
        4,
        codebook,
        DataSource::Csv(file.path().to_path_buf()),
        0,
        rng,
    )
    .unwrap();
    engine.run(200).unwrap();
}

// Smoke test satellites dataset csv which is pretty messy and sparse. This has
// caught errors not caught by other tests.
#[test]
fn satellites_csv_workflow() {
    let path = PathBuf::from("resources/datasets/satellites/data.csv");

    // default codebook
    let codebook =
        codebook_from_csv(path.as_path(), None, None, false).unwrap();

    let mut engine: Engine = EngineBuilder::new(DataSource::Csv(path))
        .codebook(codebook)
        .with_nstates(4)
        .seed_from_u64(1776)
        .build()
        .unwrap();

    let config = EngineUpdateConfig::with_default_transitions().n_iters(100);

    engine
        .update(&config, Timeout::new(Duration::from_secs(30)))
        .unwrap();
}

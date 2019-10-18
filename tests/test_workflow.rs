use braid::data::DataSource;
use braid::Engine;
use braid_codebook::csv::codebook_from_csv;
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
    csv_file.write(&csv_data.as_bytes()).unwrap();

    let csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_data.as_bytes());

    // default codebook
    let codebook = codebook_from_csv(csv_reader, None, None, None);
    let rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let mut engine =
        Engine::new(4, codebook, DataSource::Csv(path.clone()), 0, rng);
    engine.run(200);
    remove_file(path).unwrap();
}

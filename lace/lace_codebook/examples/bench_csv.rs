// use lace_codebook::csv::ReaderGenerator;
use std::path::PathBuf;
use std::time::Instant;

// use lace_codebook::csv::codebook_from_csv as csv_old;
use lace_codebook::data::codebook_from_csv as csv_new;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(args[1].as_str());

    // let reader_generator = ReaderGenerator::Csv(path.clone());

    // let now = Instant::now();
    // let _codebook =
    //     csv_old(reader_generator, None, None, false, false).unwrap();
    // let t_old = now.elapsed();

    // println!("t_old: {}s", t_old.as_secs_f64());

    let now = Instant::now();
    let _codebook = csv_new(path, None, None, None, true);
    let t_new = now.elapsed();

    println!("t_new: {}s", t_new.as_secs_f64());
}

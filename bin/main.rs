#![feature(rustc_private)]

#[macro_use]
extern crate clap;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate log;
#[macro_use]
extern crate maplit;

extern crate rayon;

mod bench;
mod geweke;
mod pit;
mod regression;
mod routes;
mod shapes;
mod utils;

use self::clap::App;

fn main() {
    let yaml = load_yaml!("cli.yaml");
    let app = App::from_yaml(yaml).get_matches();

    let verbose: bool = app.occurrences_of("verb") > 0;

    match app.subcommand() {
        ("run", Some(sub_m)) => routes::run(&sub_m, verbose),
        ("codebook", Some(sub_m)) => routes::codebook(&sub_m, verbose),
        ("bench", Some(sub_m)) => routes::bench(&sub_m, verbose),
        ("append", Some(sub_m)) => routes::append(&sub_m, verbose),
        ("regression", Some(sub_m)) => regression::regression(&sub_m, verbose),
        _ => (),
    }
}

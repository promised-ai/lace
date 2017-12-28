#[macro_use]
extern crate clap;
extern crate braid;
extern crate jsonrpc_core;
extern crate jsonrpc_http_server;
extern crate rusqlite;

use std::path::Path;
use std::str::FromStr;
use std::fmt::Debug;
use self::rusqlite::Connection;
use self::jsonrpc_core::{IoHandler, Params, Value};
use self::jsonrpc_http_server::{ServerBuilder};
use self::clap::{App, ArgMatches};



fn parse_arg<T>(arg_name: &str, matches: &ArgMatches) -> T
where T: FromStr
{
    match matches.value_of(arg_name).unwrap().parse::<T>() {
        Ok(x)  => x,
        Err(_) => panic!("Could not parse {}", arg_name),
    }
}


pub fn build_server() -> ServerBuilder {
    let mut io = IoHandler::new();
    io.add_method("version", |_: Params| {
        Ok(Value::String("Braid version 0.0.1dev".into()))
    });

    ServerBuilder::new(io)
}


use braid::cc::state::StateGewekeSettings;
use braid::geweke::GewekeTester;
use braid::cc::State;


fn run_geweke(sub_m: &ArgMatches, verbose: bool) {
    let nrows: usize = parse_arg("nrows", &sub_m);
    let ncols: usize = parse_arg("ncols", &sub_m);
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str = sub_m.value_of("output").unwrap();

    let settings = StateGewekeSettings::new(nrows, ncols);

    let mut geweke: GewekeTester<State> = GewekeTester::new(settings);
    if verbose {
        geweke.set_verbose(true);
        println!("Created GewekeTester w/ {} rows and {} cols", nrows, ncols);
    }
    geweke.run(n_iter);
    geweke.save(Path::new(output));
}


fn load_engine(sub_m: &ArgMatches, verbose: bool) {
    unimplemented!();
}


// fn read_from_yaml<T: Deserialize>(path: &str) -> T {
//     let mut file = File::open(Path::from(&cb_path)).unwrap();
//     let mut yaml = String::new();
//     let res = file.read_to_string(&mut yaml).unwrap();
//     serde_yaml::from_str(&yaml).unwrap()
// }


fn new_engine(sub_m: &ArgMatches, verbose: bool) {
    unimplemented!();
    // let db_path: &str = sub_m.value_of("sqlite").unwrap();
    // let cb_path: usize = parse_arg("codebook", &sub_m);
    // let nstates: usize = parse_arg("nstates", &sub_m);
    // let n_iter: usize = parse_arg("niter", &sub_m);
    // let checkpoint: usize = parse_arg("checkpoint", &sub_m);
    // let output: &str = sub_m.value_of("output").unwrap();

    // let codebook: Codebook = read_from_yaml(&cb_path);
    // let mut engine = Engine::from_sqlite(&db_path, &codebook, nstates);
    // engine.run(n_iter, checkpoint);
    // engine.save(output);
}


fn main() {
    // let _server = build_server()
    //     .start_http(&"127.0.0.1:2723".parse().unwrap());
    let yaml = load_yaml!("cli.yaml");
    let app = App::from_yaml(yaml).get_matches();

    let verbose: bool = app.occurrences_of("v") > 0;

    match app.subcommand() {
        ("geweke", Some(sub_m)) => run_geweke(&sub_m, verbose),
        ("new", Some(sub_m))    => new_engine(&sub_m, verbose),
        ("load", Some(sub_m))   => load_engine(&sub_m, verbose),
        _                       => (),
    }
}

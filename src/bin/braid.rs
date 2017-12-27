#[macro_use]
extern crate clap;
extern crate braid;
extern crate jsonrpc_core;
extern crate jsonrpc_http_server;

use std::path::Path;
use std::str::FromStr;
use std::fmt::Debug;
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


pub fn run_geweke(sub_m: &ArgMatches, verbose: bool) -> GewekeTester<State> {
    let nrows: usize = parse_arg("nrows", &sub_m);
    let ncols: usize = parse_arg("ncols", &sub_m);
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str = sub_m.value_of("output").unwrap();

    let settings = StateGewekeSettings::new(nrows, ncols);    

    let mut geweke = GewekeTester::new(settings);
    if verbose {
        geweke.set_verbose(true);
        println!("Created GewekeTester w/ {} rows and {} cols", nrows, ncols);
    }
    geweke.run(n_iter);
    geweke.save_results(Path::new(output));

    geweke
}


fn main() {
    // let _server = build_server()
    //     .start_http(&"127.0.0.1:2723".parse().unwrap());
    let yaml = load_yaml!("cli.yaml");
    let app = App::from_yaml(yaml).get_matches();

    let verbose: bool = app.occurrences_of("v") > 0;

    match app.subcommand() {
        ("geweke", Some(sub_m)) => {
            let geweke = run_geweke(&sub_m, verbose);
        },
        ("new", Some(sub_m))    => println!("new!"),
        ("load", Some(sub_m))   => println!("load!"),
        _                       => (),
    }
}

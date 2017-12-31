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
use braid::cc::view::ViewGewekeSettings;
use braid::geweke::{GewekeTester, GewekeModel};
use braid::cc::{State, View, ColModelType};


fn get_cm_types(sub_m: &ArgMatches, ncols: usize) -> Vec<ColModelType> {
    let mut cm_types = Vec::with_capacity(ncols);
    let use_default = sub_m.occurrences_of("coltypes") < 1;

    if use_default {
        cm_types = vec![ColModelType::Continuous; ncols];
    } else {
        let cm_flags = sub_m
            .values_of("coltypes")
            .unwrap()
            .collect::<Vec<_>>();

        let nflags = cm_flags.len(); 
        if nflags != ncols {
            panic!("Requested {} columns, but specified {} column types",
                ncols, nflags);
        }

        cm_flags.iter().for_each(|flag| {
            match flag {
                &"c" => cm_types.push(ColModelType::Continuous),
                &"t" => cm_types.push(ColModelType::Categorical),
                _   => panic!("Invalid coltype flag: {}", flag),
            }
        });
    }

    cm_types
}


fn run_geweke(sub_m: &ArgMatches, verbose: bool) {
    let nrows: usize = parse_arg("nrows", &sub_m);
    let ncols: usize = parse_arg("ncols", &sub_m);
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str = sub_m.value_of("output").unwrap();
    let view_only: bool = sub_m.occurrences_of("view-only") > 0;

    let cm_types = get_cm_types(sub_m, ncols);

    if view_only {
        if verbose { println!("Created View Geweke ({}, {})", nrows, ncols); }
        let settings = ViewGewekeSettings::new(nrows, cm_types);
        create_run_and_save_geweke::<View>(settings, n_iter, output, verbose);
    } else {
        if verbose { println!("Created State Geweke ({}, {})", nrows, ncols); }
        let settings = StateGewekeSettings::new(nrows, cm_types);
        create_run_and_save_geweke::<State>(settings, n_iter, output, verbose);
    }
}

fn create_run_and_save_geweke<G>(
    settings: G::Settings, n_iter: usize, output: &str, verbose: bool)
    where G: GewekeModel
{
    let mut geweke: GewekeTester<G> = GewekeTester::new(settings);
    if verbose {
        geweke.set_verbose(true);
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

    let verbose: bool = app.occurrences_of("verb") > 0;

    match app.subcommand() {
        ("geweke", Some(sub_m)) => run_geweke(&sub_m, verbose),
        ("new", Some(sub_m))    => new_engine(&sub_m, verbose),
        ("load", Some(sub_m))   => load_engine(&sub_m, verbose),
        _                       => (),
    }
}

#[macro_use]
extern crate clap;
extern crate braid;
extern crate rusqlite;
extern crate rand;


use std::path::Path;
use std::str::FromStr;
use self::clap::{App, ArgMatches};
use braid::Oracle;
use braid::interface::server::server::run_oracle_server;


fn parse_arg<T: FromStr>(arg_name: &str, matches: &ArgMatches) -> T {
    match matches.value_of(arg_name).unwrap().parse::<T>() {
        Ok(x)  => x,
        Err(_) => panic!("Could not parse {}", arg_name),
    }
}


use braid::cc::state::StateGewekeSettings;
use braid::cc::view::ViewGewekeSettings;
use braid::geweke::{GewekeTester, GewekeModel};
use braid::cc::{State, View, FType, Codebook};
use braid::data::{DataSource, SerializedType};


fn get_cm_types(sub_m: &ArgMatches, ncols: usize) -> Vec<FType> {
    let mut cm_types = Vec::with_capacity(ncols);
    let use_default = sub_m.occurrences_of("coltypes") < 1;

    if use_default {
        cm_types = vec![FType::Continuous; ncols];
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
                &"c" => cm_types.push(FType::Continuous),
                &"t" => cm_types.push(FType::Categorical),
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


fn new_engine(sub_m: &ArgMatches, _verbose: bool) {
    let use_sqlite: bool = sub_m.occurrences_of("sqlite_src") > 0;
    let use_csv: bool = sub_m.occurrences_of("csv_src") > 0;

    if (use_sqlite && use_csv) || !(use_sqlite || use_csv) {
        panic!("One of sqlite_src or csv_src must be specified");
    }

    let cb_path = sub_m.value_of("codebook").unwrap();
    let codebook = Codebook::from_yaml(&cb_path);

    let src_path;
    let data_source;

    if use_sqlite {
        src_path = sub_m.value_of("sqlite_src").unwrap();
        data_source = DataSource::Sqlite;
    } else if use_csv {
        src_path = sub_m.value_of("csv_src").unwrap();
        data_source = DataSource::Csv;
    } else {
        unreachable!();
    }

    let nstates: usize = parse_arg("nstates", &sub_m);
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str = sub_m.value_of("output").unwrap();
    // let checkpoint: usize = parse_arg("checkpoint", &sub_m);

    let mut engine = Oracle::new(nstates, codebook, Path::new(&src_path),
                                 data_source);

    engine.run(n_iter, 0);
    engine.save(Path::new(&output), SerializedType::MessagePack);
}


fn run_engine(sub_m: &ArgMatches, _verbose: bool) {

    let path = sub_m.value_of("path").unwrap();
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str = sub_m.value_of("output").unwrap();
    // let checkpoint: usize = parse_arg("checkpoint", &sub_m);

    let mut engine = Oracle::load(Path::new(&path), SerializedType::MessagePack);

    engine.run(n_iter, 0);
    engine.save(Path::new(&output), SerializedType::MessagePack);
}


fn run_oracle(sub_m: &ArgMatches, verbose: bool) {
    let path = sub_m.value_of("path").unwrap();
    let port = sub_m.value_of("port").unwrap();
    let oracle = Oracle::load(Path::new(&path), SerializedType::MessagePack);
    run_oracle_server(oracle, port);
}


fn main() {
    // let _server = build_server()
    //     .start_http(&"127.0.0.1:2723".parse().unwrap());
    let yaml = load_yaml!("cli.yaml");
    let app = App::from_yaml(yaml).get_matches();

    let verbose: bool = app.occurrences_of("verb") > 0;

    match app.subcommand() {
        ("geweke", Some(sub_m)) => run_geweke(&sub_m, verbose),
        ("run", Some(sub_m))    => new_engine(&sub_m, verbose),
        ("load", Some(sub_m))   => run_engine(&sub_m, verbose),
        ("oracle", Some(sub_m))   => run_oracle(&sub_m, verbose),
        _                       => (),
    }
}

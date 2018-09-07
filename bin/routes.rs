extern crate braid;
extern crate csv;
extern crate rand;
extern crate rusqlite;
extern crate rv;
extern crate serde_yaml;

use std::fs::File;
use std::io::Write;
use std::path::Path;

use self::csv::ReaderBuilder;
use self::rand::prng::XorShiftRng;
use self::rand::FromEntropy;
use clap::{ArgMatches, Values};
use utils::parse_arg;

use self::braid::cc::transition::StateTransition;
use self::braid::cc::Codebook;
use self::braid::cc::{ColAssignAlg, RowAssignAlg};
use self::braid::data::csv::codebook_from_csv;
use self::braid::data::DataSource;
use self::braid::interface::Bencher;
use self::braid::{Engine, EngineBuilder};
use self::rv::dist::Gamma;

fn get_transitions(transitions: Option<Values>) -> Vec<StateTransition> {
    if transitions.is_none() {
        braid::cc::State::default_transitions()
    } else {
        let tvec: Vec<&str> = transitions.unwrap().collect();
        tvec.iter()
            .map(|t| match &t[..] {
                "col_assign" => StateTransition::ColumnAssignment,
                "row_assign" => StateTransition::RowAssignment,
                "state_alpha" => StateTransition::StateAlpha,
                "view_alphas" => StateTransition::ViewAlphas,
                "feature_priors" => StateTransition::FeaturePriors,
                "component_params" => StateTransition::ComponentParams,
                _ => panic!("Invalid transition: {}", t),
            }).collect()
    }
}

fn get_row_and_col_algs(sub_m: &ArgMatches) -> (RowAssignAlg, ColAssignAlg) {
    let row_assign_alg = match sub_m.value_of("row_alg") {
        Some("finite-cpu") => RowAssignAlg::FiniteCpu,
        Some("gibbs") => RowAssignAlg::Gibbs,
        Some("slice") => RowAssignAlg::Slice,
        _ => panic!("Invalid row-alg"),
    };

    let col_assign_alg = match sub_m.value_of("col_alg") {
        Some("finite-cpu") => ColAssignAlg::FiniteCpu,
        Some("gibbs") => ColAssignAlg::Gibbs,
        Some("slice") => ColAssignAlg::Slice,
        _ => panic!("Invalid col-alg"),
    };

    (row_assign_alg, col_assign_alg)
}

fn new_engine(sub_m: &ArgMatches, _verbose: bool) {
    let use_sqlite: bool = sub_m.occurrences_of("sqlite_src") > 0;
    let use_csv: bool = sub_m.occurrences_of("csv_src") > 0;

    if (use_sqlite && use_csv) || !(use_sqlite || use_csv) {
        panic!("One of sqlite_src or csv_src must be specified");
    }

    let codebook_opt = match sub_m.value_of("codebook") {
        Some(cb_path) => Some(Codebook::from_yaml(&cb_path)),
        None => None,
    };

    let data_source = if use_sqlite {
        let src_path = sub_m.value_of("sqlite_src").unwrap();
        DataSource::Sqlite(String::from(src_path))
    } else if use_csv {
        let src_path = sub_m.value_of("csv_src").unwrap();
        DataSource::Csv(String::from(src_path))
    } else {
        unreachable!();
    };

    let transitions_vec = sub_m.values_of("transitions");
    let transitions = get_transitions(transitions_vec);

    let nstates: usize = parse_arg("nstates", &sub_m);
    let id_offset: usize = parse_arg("id_offset", &sub_m);
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str = sub_m.value_of("output").unwrap();
    let (row_assign_alg, col_assign_alg) = get_row_and_col_algs(&sub_m);

    let mut builder = EngineBuilder::new(data_source)
        .with_nstates(nstates)
        .with_id_offset(id_offset);

    builder = match codebook_opt {
        Some(codebook) => builder.with_codebook(codebook),
        None => builder,
    };

    let mut engine = builder.build().expect("Failed to build Engine.");

    engine.update(n_iter, row_assign_alg, col_assign_alg, transitions, false);
    engine
        .save(&output)
        .expect("Failed to save. I'm really sorry.");
}

fn run_engine(sub_m: &ArgMatches, _verbose: bool) {
    let path = sub_m.value_of("engine").expect("no 'engine' supplied.");
    let n_iter: usize = parse_arg("niter", &sub_m);
    let output: &str =
        sub_m.value_of("output").expect("no output path supplied");
    let (row_assign_alg, col_assign_alg) = get_row_and_col_algs(&sub_m);

    let mut engine = Engine::load(&path).expect("could not load engine.");

    let transitions_vec = sub_m.values_of("transitions");
    let transitions = get_transitions(transitions_vec);

    engine.update(n_iter, row_assign_alg, col_assign_alg, transitions, false);
    engine
        .save(&output)
        .expect("failed to save. i'm really sorry.");
}

pub fn run(sub_m: &ArgMatches, _verbose: bool) {
    if sub_m.occurrences_of("engine") > 0 {
        run_engine(sub_m, _verbose);
    } else {
        new_engine(sub_m, _verbose);
    }
}

pub fn codebook(sub_m: &ArgMatches, _verbose: bool) {
    let path_in = sub_m.value_of("csv_src").unwrap();
    let path_out = sub_m.value_of("output").unwrap();
    let alpha_prior = match sub_m.values_of("alpha_prior") {
        Some(wrapper) => {
            let params: Vec<&str> = wrapper.collect();
            let shape: f64 = params[0].parse().unwrap();
            let rate: f64 = params[1].parse().unwrap();
            let prior =
                Gamma::new(shape, rate).expect("Invalid alpha prior params");
            Some(prior)
        }
        None => None,
    };

    let reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(&path_in))
        .unwrap();

    let gmd_reader = match sub_m.value_of("genomic_metadata") {
        Some(dir) => {
            let r = ReaderBuilder::new()
                .has_headers(true)
                .from_path(Path::new(&dir))
                .unwrap();
            Some(r)
        }
        None => None,
    };

    let codebook = codebook_from_csv(reader, None, alpha_prior, gmd_reader);
    let bytes = serde_yaml::to_string(&codebook).unwrap().into_bytes();

    let path_out = Path::new(&path_out);
    let mut file = File::create(path_out).unwrap();
    file.write(&bytes).unwrap();
    println!("Wrote file {:?}", path_out);
    println!("Always be sure to verify the codebook");
}

pub fn bench(sub_m: &ArgMatches, _verbose: bool) {
    let path_string = String::from(sub_m.value_of("csv_src").unwrap());
    let (row_assign_alg, col_assign_alg) = get_row_and_col_algs(&sub_m);
    let n_iters: usize = parse_arg("n_iters", &sub_m);
    let n_runs: usize = parse_arg("n_runs", &sub_m);

    let reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(&path_string))
        .unwrap();

    let codebook = codebook_from_csv(reader, None, None, None);

    let bencher = Bencher::from_csv(codebook, path_string)
        .with_n_iters(n_iters)
        .with_n_runs(n_runs)
        .with_col_assign_alg(col_assign_alg)
        .with_row_assign_alg(row_assign_alg);

    let mut rng = XorShiftRng::from_entropy();
    let results = bencher.run(&mut rng);

    let res_string = serde_yaml::to_string(&results).unwrap();
    println!("{}", res_string);
}

pub fn append(sub_m: &ArgMatches, _verbose: bool) {
    let use_sqlite: bool = sub_m.occurrences_of("sqlite_src") > 0;
    let use_csv: bool = sub_m.occurrences_of("csv_src") > 0;
    let output: &str = sub_m
        .value_of("output")
        .expect("Output braidfile path required.");
    let input: &str = sub_m
        .value_of("input")
        .expect("Input braidfile path required.");

    if (use_sqlite && use_csv) || !(use_sqlite || use_csv) {
        panic!("One of sqlite_src or csv_src must be specified");
    }

    let data_source = if use_sqlite {
        let src_path = sub_m.value_of("sqlite_src").unwrap();
        DataSource::Sqlite(String::from(src_path))
    } else if use_csv {
        let src_path = sub_m.value_of("csv_src").unwrap();
        DataSource::Csv(String::from(src_path))
    } else {
        unreachable!();
    };

    let codebook: Codebook = match sub_m.value_of("codebook") {
        Some(cb_path) => Codebook::from_yaml(&cb_path),
        None => data_source.default_codebook().unwrap(),
    };

    // If codebook not supplied, make one
    let mut engine = Engine::load(&input).expect("Could not load engine.");
    engine.append_features(codebook, data_source);
    engine.save(&output).expect("Could not save engine.")
}

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
use braid_opt;

use self::braid::cc::config::EngineUpdateConfig;
use self::braid::cc::Codebook;
use self::braid::data::csv::codebook_from_csv;
use self::braid::data::DataSource;
use self::braid::interface::Bencher;
use self::braid::{Engine, EngineBuilder};
use self::rv::dist::Gamma;

fn new_engine(cmd: braid_opt::RunCmd) -> i32 {
    let use_sqlite: bool = cmd.sqlite_src.is_some();
    let use_csv: bool = cmd.csv_src.is_some();

    if (use_sqlite && use_csv) || !(use_sqlite || use_csv) {
        panic!("One of sqlite_src or csv_src must be specified");
    }

    let codebook_opt = match cmd.codebook {
        Some(cb_path) => Some(Codebook::from_yaml(&cb_path)),
        None => None,
    };

    let data_source = if use_sqlite {
        DataSource::Sqlite(cmd.sqlite_src.unwrap())
    } else if use_csv {
        DataSource::Csv(cmd.csv_src.unwrap())
    } else {
        unreachable!();
    };

    let mut builder = EngineBuilder::new(data_source)
        .with_nstates(cmd.nstates)
        .with_id_offset(cmd.id_offset);

    builder = match codebook_opt {
        Some(codebook) => builder.with_codebook(codebook),
        None => builder,
    };

    let mut engine = match builder.build() {
        Ok(engine) => engine,
        Err(..) => {
            eprintln!("Failed to build engine");
            return 1;
        }
    };

    let config = EngineUpdateConfig::new()
        .with_iters(cmd.n_iters)
        .with_timeout(cmd.timeout)
        .with_row_alg(cmd.row_alg)
        .with_col_alg(cmd.col_alg)
        .with_transitions(cmd.transitions);

    engine.update(config);
    if engine.save(&cmd.output).is_ok() {
        0
    } else {
        eprintln!("Failed to save.");
        1
    }
}

fn run_engine(cmd: braid_opt::RunCmd) -> i32 {
    let mut engine = match Engine::load(&cmd.engine.unwrap()) {
        Ok(engine) => engine,
        Err(..) => {
            eprintln!("Could not load engine");
            return 1;
        }
    };

    let config = EngineUpdateConfig::new()
        .with_iters(cmd.n_iters)
        .with_timeout(cmd.timeout)
        .with_row_alg(cmd.row_alg)
        .with_col_alg(cmd.col_alg)
        .with_transitions(cmd.transitions);

    engine.update(config);
    if engine.save(&cmd.output).is_ok() {
        0
    } else {
        eprintln!("Failed to save.");
        1
    }
}

pub fn run(cmd: braid_opt::RunCmd) -> i32 {
    if cmd.engine.is_some() {
        run_engine(cmd)
    } else {
        new_engine(cmd)
    }
}

pub fn codebook(cmd: braid_opt::CodebookCmd) -> i32 {
    let alpha_prior =
        Some(Gamma::new(cmd.alpha_prior.a, cmd.alpha_prior.b).unwrap());

    if !Path::new(cmd.csv_src.as_str()).exists() {
        eprintln!("CSV input {} not found", cmd.csv_src);
        return 1;
    }

    let reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(&cmd.csv_src))
        .unwrap();

    let gmd_reader = match cmd.genomic_metadata {
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

    let path_out = Path::new(&cmd.output);
    let mut file = File::create(path_out).unwrap();
    file.write(&bytes).unwrap();
    println!("Wrote file {:?}", path_out);
    println!("Always be sure to verify the codebook");

    0
}

pub fn bench(cmd: braid_opt::BenchCmd) -> i32 {
    let reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(&cmd.csv_src))
        .unwrap();

    let codebook = codebook_from_csv(reader, None, None, None);

    let bencher = Bencher::from_csv(codebook, cmd.csv_src)
        .with_n_iters(cmd.n_iters)
        .with_n_runs(cmd.n_runs)
        .with_col_assign_alg(cmd.col_alg)
        .with_row_assign_alg(cmd.row_alg);

    let mut rng = XorShiftRng::from_entropy();
    let results = bencher.run(&mut rng);

    let res_string = serde_yaml::to_string(&results).unwrap();
    println!("{}", res_string);

    0
}

pub fn append(cmd: braid_opt::AppendCmd) -> i32 {
    let use_sqlite: bool = cmd.sqlite_src.is_some();
    let use_csv: bool = cmd.csv_src.is_some();

    let output: &str = cmd.output.as_str();
    let input: &str = cmd.input.as_str();

    if (use_sqlite && use_csv) || !(use_sqlite || use_csv) {
        panic!("One of sqlite_src or csv_src must be specified");
    }

    let data_source = if use_sqlite {
        DataSource::Sqlite(cmd.sqlite_src.unwrap())
    } else if use_csv {
        let src = cmd.csv_src.unwrap();

        if Path::new(src.as_str()).exists() {
            DataSource::Csv(src)
        } else {
            println!("CSV input {} does not exist.", { src });
            return 1;
        }
    } else {
        unreachable!();
    };

    let codebook: Codebook = match cmd.codebook {
        Some(cb_path) => Codebook::from_yaml(cb_path.as_str()),
        None => data_source.default_codebook().unwrap(),
    };

    // If codebook not supplied, make one
    let mut engine = Engine::load(&input).expect("Could not load engine.");
    engine.append_features(codebook, data_source);
    engine.save(&output).expect("Could not save engine.");

    0
}

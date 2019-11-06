use std::fs::File;
use std::io::Write;
use std::path::Path;

use braid::cc::config::EngineUpdateConfig;
use braid::data::DataSource;
use braid::file_config::SerializedType;
use braid::interface::{Bencher, Engine, EngineBuilder};

use braid_codebook::codebook::Codebook;
use braid_codebook::csv::codebook_from_csv;
use csv::ReaderBuilder;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::braid_opt;

pub fn summarize_engine(cmd: braid_opt::SummarizeCmd) -> i32 {
    use prettytable::{cell, format, row, Table};

    let engine = match Engine::load(cmd.braidfile.as_path()) {
        Ok(engine) => engine,
        Err(..) => {
            eprintln!("Could not load engine");
            return 1;
        }
    };

    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_CLEAN);
    table.add_row(
        row![b->"State", b->"Iters", b->"Views", b->"Alpha", b->"Score"],
    );
    for (id, state) in engine.states {
        let diag = &state.diagnostics;
        let n = diag.nviews.len() - 1;
        table.add_row(row![
            format!("{}", id),
            format!("{}", n + 1),
            format!("{}", diag.nviews[n]),
            format!("{}", diag.state_alpha[n]),
            format!("{}", diag.loglike[n]),
        ]);
    }
    table.printstd();
    0
}

fn new_engine(cmd: braid_opt::RunCmd) -> i32 {
    // XXX: It might look like we could supply both a sqlite and a csv source,
    // but the structopts setup won't allow it, so don't worry
    let use_sqlite: bool = cmd.sqlite_src.is_some();
    let use_csv: bool = cmd.csv_src.is_some();

    let codebook_opt = match cmd.codebook {
        Some(cb_path) => Some(Codebook::from_yaml(&cb_path.as_path()).unwrap()),
        None => None,
    };

    let data_source = if use_sqlite {
        DataSource::Sqlite(cmd.sqlite_src.unwrap())
    } else if use_csv {
        DataSource::Csv(cmd.csv_src.unwrap())
    } else {
        eprintln!("No data source provided.");
        return 1;
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

    let save_result = engine
        .save_to(&cmd.output)
        // .with_serialized_type(SerializedType::Bincode)
        .with_serialized_type(SerializedType::Yaml)
        .save();

    match save_result {
        Ok(..) => 0,
        Err(err) => {
            eprintln!("Failed to save: {:?}", err);
            1
        }
    }
}

fn run_engine(cmd: braid_opt::RunCmd) -> i32 {
    let engine_dir = cmd.engine.unwrap();
    let mut engine = match Engine::load(&engine_dir) {
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
    let save_result = engine
        .save_to(&cmd.output)
        .with_serialized_type(SerializedType::Bincode)
        .save();

    if save_result.is_ok() {
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
    if !cmd.csv_src.exists() {
        eprintln!("CSV input {:?} not found", cmd.csv_src);
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

    let codebook = codebook_from_csv(
        reader,
        Some(cmd.category_cutoff),
        Some(cmd.alpha_prior),
        gmd_reader,
    );
    let bytes = serde_yaml::to_string(&codebook).unwrap().into_bytes();

    let path_out = Path::new(&cmd.output);
    let mut file = File::create(path_out).unwrap();
    file.write_all(&bytes).unwrap();
    println!("Wrote file {:?}", path_out);
    println!("Always be sure to verify the codebook");

    0
}

pub fn bench(cmd: braid_opt::BenchCmd) -> i32 {
    let reader = match ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(&cmd.csv_src))
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Could not read csv {:?}. {}", cmd.csv_src, e);
            return 1;
        }
    };

    let codebook = codebook_from_csv(reader, None, None, None);

    let bencher = Bencher::from_csv(codebook, cmd.csv_src)
        .with_n_iters(cmd.n_iters)
        .with_n_runs(cmd.n_runs)
        .with_col_assign_alg(cmd.col_alg)
        .with_row_assign_alg(cmd.row_alg);

    let mut rng = Xoshiro256Plus::from_entropy();
    let results = bencher.run(&mut rng);

    let res_string = serde_yaml::to_string(&results).unwrap();
    println!("{}", res_string);

    0
}

fn append_columns(cmd: braid_opt::AppendCmd) -> Result<Engine, i32> {
    let use_sqlite: bool = cmd.sqlite_src.is_some();
    let use_csv: bool = cmd.csv_src.is_some();

    if (use_sqlite && use_csv) || !(use_sqlite || use_csv) {
        panic!("One of sqlite_src or csv_src must be specified");
    }

    let data_source = if use_sqlite {
        DataSource::Sqlite(cmd.sqlite_src.unwrap())
    } else if use_csv {
        let src = cmd.csv_src.unwrap();

        if src.exists() {
            DataSource::Csv(src)
        } else {
            println!("CSV input {:?} does not exist.", { src });
            return Err(1);
        }
    } else {
        unreachable!();
    };

    let codebook: Codebook = match cmd.codebook {
        Some(cb_path) => Codebook::from_yaml(cb_path).unwrap(),
        None => data_source.default_codebook().unwrap(),
    };

    // If codebook not supplied, make one
    let mut engine = Engine::load(&cmd.input).expect("Could not load engine.");
    engine.append_features(codebook, data_source);

    Ok(engine)
}

fn append_rows(cmd: braid_opt::AppendCmd) -> Result<Engine, i32> {
    let data_source = match cmd.csv_src {
        Some(path) => DataSource::Csv(path),
        None => return Err(1),
    };
    let mut engine = Engine::load(&cmd.input).expect("Could not load engine.");
    engine.append_rows(data_source);
    Ok(engine)
}

pub fn append(cmd: braid_opt::AppendCmd) -> i32 {
    let output = cmd.output.clone();
    let engine = if cmd.columns {
        append_columns(cmd).unwrap()
    } else {
        append_rows(cmd).unwrap()
    };

    let save_result = engine
        .save_to(&output)
        .with_serialized_type(SerializedType::Bincode)
        .save();

    if save_result.is_ok() {
        0
    } else {
        eprintln!("Failed to save.");
        1
    }
}

pub fn regen_examples() -> i32 {
    use braid::examples::Example;

    println!("Regenerating Animals metadata...");
    if let Err(err) = Example::Animals.regen_metadata() {
        eprintln!("Error running Animals: {:?}", err);
        1
    } else {
        println!("Done.");
        0
    }
}

use std::io::Write;
use std::time::Duration;

use lace::codebook::Codebook;
use lace::metadata::{deserialize_file, serialize_obj};
use lace::stats::rv::dist::Gamma;
use lace::update_handler::{CtrlC, ProgressBar, Timeout};
use lace::{Builder, Engine};

use crate::opt;

pub fn summarize_engine(cmd: opt::SummarizeArgs) -> i32 {
    let load_res = Engine::load(cmd.lacefile.as_path());
    let engine = match load_res {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Could not load engine: {err:?}");
            return 1;
        }
    };

    let header = vec![
        String::from("State"),
        String::from("Iters"),
        String::from("Views"),
        String::from("Alpha"),
        String::from("Score"),
    ];

    let mut rows: Vec<Vec<String>> = engine
        .state_ids
        .iter()
        .zip(engine.states.iter())
        .map(|(id, state)| {
            let diag = &state.diagnostics;
            let n = diag.n_views.len() - 1;
            vec![
                format!("{id}"),
                format!("{}", n + 1),
                format!("{}", diag.n_views[n]),
                format!("{:.6}", diag.state_alpha[n]),
                format!("{:.6}", diag.loglike[n]),
            ]
        })
        .collect();

    rows.sort_by_key(|row| row[0].clone());

    crate::utils::print_table(header, rows);
    0
}

fn new_engine(cmd: opt::RunArgs) -> i32 {
    let mut update_config = cmd.engine_update_config();
    let save_config = cmd.file_config().unwrap();

    if update_config.save_config.is_none() {
        let config = lace::config::SaveEngineConfig {
            path: cmd.output.clone(),
            file_config: save_config.clone(),
        };
        update_config.save_config = Some(config);
        update_config.checkpoint = cmd.checkpoint;
    };

    // turn off mutability
    let update_config = update_config;
    let save_config = save_config;

    let codebook_opt: Option<Codebook> = cmd
        .codebook
        .as_ref()
        .map(|cb_path| deserialize_file(cb_path).unwrap());

    let data_source = if let Some(src) = cmd.data_source() {
        src
    } else {
        eprintln!("No data source provided.");
        return 1;
    };

    let mut builder = Builder::new(data_source)
        .with_nstates(cmd.nstates)
        .id_offset(cmd.id_offset);

    builder = match codebook_opt {
        Some(codebook) => builder.codebook(codebook),
        None => builder,
    };

    if cmd.flat_cols {
        builder = builder.flat_cols();
    }

    builder = match cmd.seed {
        Some(seed) => builder.seed_from_u64(seed),
        None => builder,
    };

    let mut engine = match builder.build() {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Failed to build engine: {e:?}");
            return 1;
        }
    };

    if cmd.quiet {
        engine.update(update_config, CtrlC::new()).unwrap();
    } else {
        engine
            .update(update_config, (ProgressBar::new(), CtrlC::new()))
            .unwrap();
    }

    eprint!("Saving...");
    std::io::stdout().flush().expect("Could not flush stdout");
    let save_result = engine.save(&cmd.output, &save_config);
    eprintln!("Done");

    match save_result {
        Ok(..) => 0,
        Err(err) => {
            eprintln!("Failed to save: {err:?}");
            1
        }
    }
}

fn run_engine(cmd: opt::RunArgs) -> i32 {
    let mut update_config = cmd.engine_update_config();
    let save_config = cmd.file_config().unwrap();

    let engine_dir = cmd.engine.clone().unwrap();

    let load_res = Engine::load(&engine_dir);
    let mut engine = match load_res {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Could not load engine: {err}");
            return 1;
        }
    };

    if update_config.save_config.is_none() {
        let config = lace::config::SaveEngineConfig {
            path: cmd.output.clone(),
            file_config: save_config.clone(),
        };
        update_config.save_config = Some(config);
        update_config.checkpoint = cmd.checkpoint;
    };

    // create timeout update handler
    let timeout = Timeout::new(
        cmd.timeout
            .map(Duration::from_secs)
            .unwrap_or(Duration::MAX),
    );

    // turn off mutability
    let save_config = save_config;
    let update_config = update_config;

    if cmd.quiet {
        engine
            .update(update_config, (timeout, CtrlC::new()))
            .unwrap();
    } else {
        engine
            .update(update_config, (timeout, ProgressBar::new(), CtrlC::new()))
            .unwrap();
    }

    eprint!("Saving...");
    std::io::stdout().flush().expect("Could not flush stdout");
    let save_result = engine.save(&cmd.output, &save_config);

    eprintln!("Done");

    if save_result.is_ok() {
        0
    } else {
        eprintln!("Failed to save.");
        1
    }
}

pub fn run(cmd: opt::RunArgs) -> i32 {
    if cmd.engine.is_some() {
        run_engine(cmd)
    } else {
        new_engine(cmd)
    }
}

macro_rules! codebook_from {
    ($path: ident, $fn: ident, $cmd: ident) => {{
        let alpha_prior: Gamma = match $cmd.alpha_prior.try_into() {
            Ok(gamma) => gamma,
            Err(err) => {
                eprint!("Invalid Gamma parameters to CRP prior: {err}");
                return 1;
            }
        };
        if !$path.exists() {
            eprintln!("Input {:?} not found", $path);
            return 1;
        }

        let codebook = match lace::codebook::data::$fn(
            $path,
            Some($cmd.category_cutoff),
            Some(alpha_prior),
            $cmd.no_hyper,
        ) {
            Ok(codebook) => codebook,
            Err(err) => {
                eprintln!("Error: {err}");
                return 1;
            }
        };
        codebook
    }};
}

pub fn codebook(cmd: opt::CodebookArgs) -> i32 {
    let codebook = if let Some(path) = cmd.csv_src {
        codebook_from!(path, codebook_from_csv, cmd)
    } else if let Some(path) = cmd.json_src {
        codebook_from!(path, codebook_from_json, cmd)
    } else if let Some(path) = cmd.parquet_src {
        codebook_from!(path, codebook_from_parquet, cmd)
    } else if let Some(path) = cmd.ipc_src {
        codebook_from!(path, codebook_from_ipc, cmd)
    } else {
        eprintln!("No source provided");
        return 1;
    };

    let res = serialize_obj(&codebook, cmd.output.as_path());

    if let Err(err) = res {
        eprintln!("Error: {err}");
        return 1;
    }

    println!("Wrote file {:?}", cmd.output);
    println!("Always be sure to verify the codebook");

    0
}

pub fn regen_examples(cmd: opt::RegenExamplesArgs) -> i32 {
    use lace::examples::Example;
    let n_iters = cmd.n_iters;
    let timeout = cmd.timeout;

    cmd.examples
        .unwrap_or_else(|| vec![Example::Animals, Example::Satellites])
        .iter()
        .try_for_each(|example| {
            println!("Regenerating {example:?} metadata...");
            if let Err(err) = example.regen_metadata(n_iters, timeout) {
                eprintln!("Error running {example:?}, {err:?}");
                Err(())
            } else {
                println!("Done.");
                Ok(())
            }
        })
        .map_or(1i32, |_| 0i32)
}

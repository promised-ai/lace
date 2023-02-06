use std::convert::TryInto;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(feature = "dev")]
use braid::bencher::Bencher;
use braid::codebook::data::codebook_from_csv;
use braid::codebook::Codebook;
use braid::metadata::{deserialize_file, serialize_obj};
use braid::{Builder, Engine};

#[cfg(feature = "dev")]
use rand::SeedableRng;
#[cfg(feature = "dev")]
use rand_xoshiro::Xoshiro256Plus;

use crate::opt;
use crate::opt::HasUserInfo;

pub fn summarize_engine(cmd: opt::SummarizeArgs) -> i32 {
    let user_info = match cmd.user_info() {
        Ok(user_info) => user_info,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let key = user_info.encryption_key().unwrap();
    let load_res = Engine::load(cmd.braidfile.as_path(), key.as_ref());
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

async fn new_engine(cmd: opt::RunArgs) -> i32 {
    let mut update_config = cmd.engine_update_config();
    let save_config = cmd.save_config().unwrap();

    if update_config.save_config.is_none() {
        let config = braid::config::SaveEngineConfig {
            path: cmd.output.clone(),
            save_config: save_config.clone(),
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

    let (sender, reciever) = braid::create_comms();
    let quit_now = Arc::new(AtomicBool::new(false));
    let quit_now_b = quit_now.clone();

    let progress = if cmd.quiet {
        None
    } else {
        Some(braid::misc::progress_bar(
            update_config.n_iters * cmd.nstates,
            reciever,
        ))
    };

    ctrlc::set_handler(move || {
        quit_now.store(true, Ordering::SeqCst);
        println!("Recieved abort.");
    })
    .expect("Error setting Ctrl-C handler");

    let run_cmd = tokio::spawn(async move {
        engine
            .update(update_config, Some(sender), Some(quit_now_b))
            .unwrap();
        engine
    });

    let _rcvr = if let Some(pbar) = progress {
        Some(pbar.await.expect("Failed to join ProgressBar"))
    } else {
        None
    };

    let save_result = run_cmd
        .await
        .map(|engine| {
            eprint!("Saving...");
            std::io::stdout().flush().expect("Could not flush stdout");
            engine.save(&cmd.output, &save_config)
        })
        .expect("Failed to join Engine::update");
    eprintln!("Done");

    match save_result {
        Ok(..) => 0,
        Err(err) => {
            eprintln!("Failed to save: {err:?}");
            1
        }
    }
}

async fn run_engine(cmd: opt::RunArgs) -> i32 {
    let mut update_config = cmd.engine_update_config();
    let save_config = cmd.save_config().unwrap();

    let engine_dir = cmd.engine.clone().unwrap();

    let key = save_config.user_info.encryption_key().unwrap();
    let load_res = Engine::load(&engine_dir, key.as_ref());
    let mut engine = match load_res {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Could not load engine: {err}");
            return 1;
        }
    };

    if update_config.save_config.is_none() {
        let config = braid::config::SaveEngineConfig {
            path: cmd.output.clone(),
            save_config: save_config.clone(),
        };
        update_config.save_config = Some(config);
        update_config.checkpoint = cmd.checkpoint;
    };

    // turn off mutability
    let save_config = save_config;
    let update_config = update_config;

    let (sender, reciever) = braid::create_comms();
    let quit_now = Arc::new(AtomicBool::new(false));
    let quit_now_b = quit_now.clone();

    let progress = if cmd.quiet {
        None
    } else {
        Some(braid::misc::progress_bar(
            update_config.n_iters * engine.n_states(),
            reciever,
        ))
    };

    ctrlc::set_handler(move || {
        quit_now.store(true, Ordering::SeqCst);
        eprintln!("Recieved abort.");
    })
    .expect("Error setting Ctrl-C handler");

    let run_cmd = tokio::spawn(async move {
        engine
            .update(update_config, Some(sender), Some(quit_now_b))
            .unwrap();
        engine
    });

    let _rcvr = if let Some(pbar) = progress {
        Some(pbar.await.expect("Failed to join ProgressBar"))
    } else {
        None
    };

    let save_result = run_cmd
        .await
        .map(|engine| {
            eprint!("Saving...");
            std::io::stdout().flush().expect("Could not flush stdout");
            engine.save(&cmd.output, &save_config)
        })
        .expect("Failed to join Engine::update");

    eprintln!("Done");

    if save_result.is_ok() {
        0
    } else {
        eprintln!("Failed to save.");
        1
    }
}

pub async fn run(cmd: opt::RunArgs) -> i32 {
    if cmd.engine.is_some() {
        run_engine(cmd).await
    } else {
        new_engine(cmd).await
    }
}

macro_rules! codebook_from {
    ($path: ident, $fn: ident, $cmd: ident) => {{
        if !$path.exists() {
            eprintln!("Input {:?} not found", $path);
            return 1;
        }

        let codebook = match braid::codebook::data::$fn(
            $path,
            Some($cmd.category_cutoff),
            Some($cmd.alpha_prior),
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

#[cfg(feature = "dev")]
pub fn bench(cmd: opt::BenchArgs) -> i32 {
    match codebook_from_csv(&cmd.csv_src, None, None, false) {
        Ok(codebook) => {
            let mut bencher = Bencher::from_csv(codebook, cmd.csv_src)
                .n_iters(cmd.n_iters)
                .n_runs(cmd.n_runs)
                .col_assign_alg(cmd.col_alg)
                .row_assign_alg(cmd.row_alg);

            let mut rng = Xoshiro256Plus::from_entropy();
            let results = bencher.run(&mut rng);

            let res_string = serde_yaml::to_string(&results).unwrap();
            println!("{res_string}");

            0
        }
        Err(err) => {
            eprintln!("Failed to construct codebook: {err:?}");
            1
        }
    }
}

#[cfg(feature = "dev")]
pub fn regen_examples(cmd: opt::RegenExamplesArgs) -> i32 {
    use braid::examples::Example;
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

pub fn keygen() -> i32 {
    // generate a 32-byte key and output in hex
    // Using rand here instead of ring, means that we do not need ring as a
    // dependency in for the top-level braid crate.
    // NOTE: According to the rand crate documentation rand::random is shorthand
    // for thread_rand().gen(). thread_rang uses ThreadRng, which uses the same
    // RNG as StdRand, which according to the docs uses the secure ChaCha12
    // generator. For more information see:
    // https://rust-random.github.io/rand/rand/rngs/struct.ThreadRng.html
    let shared_key: [u8; 32] = (0..32)
        .map(|_| rand::random::<u8>())
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap();
    let key = braid_metadata::EncryptionKey::from(shared_key);
    println!("{key}");
    0
}

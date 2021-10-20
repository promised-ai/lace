use std::fs::File;
use std::io::Write;
use std::path::Path;

use braid::bencher::Bencher;
use braid::data::DataSource;
use braid::{Engine, EngineBuilder};
use braid_codebook::csv::codebook_from_csv;
use braid_codebook::Codebook;
use csv::ReaderBuilder;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::opt;
use crate::opt::HasUserInfo;

pub fn summarize_engine(cmd: opt::SummarizeArgs) -> i32 {
    let mut user_info = match cmd.user_info() {
        Ok(user_info) => user_info,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let key = user_info.encryption_key().unwrap();
    let load_res = Engine::load(cmd.braidfile.as_path(), key);
    let engine = match load_res {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Could not load engine: {:?}", e);
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
                format!("{}", id),
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
    let use_csv: bool = cmd.csv_src.is_some();

    let update_config = cmd.engine_update_config();
    let save_config = cmd.save_config().unwrap();

    let codebook_opt = cmd
        .codebook
        .map(|cb_path| Codebook::from_yaml(&cb_path.as_path()).unwrap());

    let data_source = if use_csv {
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

    builder = match cmd.seed {
        Some(seed) => builder.with_seed(seed),
        None => builder,
    };

    let mut engine = match builder.build() {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Failed to build engine: {:?}", e);
            return 1;
        }
    };

    engine.update(update_config);

    println!("saving");
    let save_result = engine.save(&cmd.output, save_config);

    match save_result {
        Ok(..) => 0,
        Err(err) => {
            eprintln!("Failed to save: {:?}", err);
            1
        }
    }
}

fn run_engine(cmd: opt::RunArgs) -> i32 {
    let update_config = cmd.engine_update_config();
    let mut save_config = cmd.save_config().unwrap();

    let engine_dir = cmd.engine.unwrap();

    println!("load");
    let key = save_config.user_info.encryption_key().unwrap();
    let load_res = Engine::load(&engine_dir, key);
    let mut engine = match load_res {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Could not load engine: {}", err);
            return 1;
        }
    };

    engine.update(update_config);

    println!("saving");
    let save_result = engine.save(&cmd.output, save_config);

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

pub fn codebook(cmd: opt::CodebookArgs) -> i32 {
    if !cmd.csv_src.exists() {
        eprintln!("CSV input {:?} not found", cmd.csv_src);
        return 1;
    }

    let reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(&cmd.csv_src))
        .unwrap();

    let codebook: Codebook = codebook_from_csv(
        reader,
        Some(cmd.category_cutoff),
        Some(cmd.alpha_prior),
        !cmd.no_checks,
    )
    .unwrap();

    let bytes = serde_yaml::to_string(&codebook).unwrap().into_bytes();

    let path_out = Path::new(&cmd.output);
    let mut file = File::create(path_out).unwrap();
    file.write_all(&bytes).unwrap();
    println!("Wrote file {:?}", path_out);
    println!("Always be sure to verify the codebook");

    0
}

pub fn bench(cmd: opt::BenchArgs) -> i32 {
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

    match codebook_from_csv(reader, None, None, true) {
        Ok(codebook) => {
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
        Err(err) => {
            eprintln!("Failed to construct codebook: {:?}", err);
            1
        }
    }
}

pub fn regen_examples(cmd: opt::RegenExamplesArgs) -> i32 {
    use braid::examples::Example;
    let n_iters = cmd.n_iters;
    let timeout = cmd.timeout;

    cmd.examples
        .unwrap_or_else(|| vec![Example::Animals, Example::Satellites])
        .iter()
        .try_for_each(|example| {
            println!("Regenerating {:?} metadata...", example);
            if let Err(err) = example.regen_metadata(n_iters, timeout) {
                eprintln!("Error running {:?}, {:?}", example, err);
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
    let shared_key: Vec<u8> = (0..32).map(|_| rand::random::<u8>()).collect();
    let key_string = hex::encode(shared_key.as_slice());
    println!("{}", key_string);
    0
}

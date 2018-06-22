#![feature(rustc_private)]
#![feature(assoc_unix_epoch)]

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate log;

extern crate braid;
extern crate clap;
extern crate env_logger;
extern crate rand;
extern crate regex;
extern crate serde_json;
extern crate serde_yaml;

mod bench;
mod geweke;
mod ppc;
mod shapes;

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;

use self::clap::{App, Arg};
use self::rand::prng::XorShiftRng;
use self::rand::SeedableRng;
use self::regex::Regex;

use bench::{run_benches, BenchmarkRegressionConfig, BenchmarkResult};
use geweke::{run_geweke, GewekeRegressionConfig, GewekeRegressionResult};
use ppc::{run_ppc, PpcDistance, PpcRegressionConfig};
use shapes::{run_shapes, ShapeResult, ShapesRegressionConfig};

/// Configuration for regression testing
#[derive(Serialize, Deserialize)]
struct RegressionConfig {
    id: String,
    #[serde(default)]
    ppc: Option<PpcRegressionConfig>,
    #[serde(default)]
    geweke: Option<GewekeRegressionConfig>,
    #[serde(default)]
    shapes: Option<ShapesRegressionConfig>,
    #[serde(default)]
    benchmark: Option<BenchmarkRegressionConfig>,
}

/// Regression test results
#[derive(Serialize)]
struct RegressionResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    shapes: Option<Vec<ShapeResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    benchmark: Option<BenchmarkResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    geweke: Option<GewekeRegressionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ppc: Option<BTreeMap<String, Vec<PpcDistance>>>,
    run_info: RegressionRunInfo,
}

#[derive(Serialize)]
struct RegressionRunInfo {
    /// Unix timestamp
    timestamp: u64,
    /// lscpu output
    cpu_info: String,
    /// Git commit hash
    git_hash: String,
    /// Git commit message
    git_log: String,
}

impl RegressionRunInfo {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Returns the commit hash then a space then the commit message
        let git_cmd = Command::new("git")
            .arg("log")
            .arg("-n 1")
            .arg("--pretty=oneline")
            .output()
            .expect("Failed to execute `git log -n 1 --pretty=oneline`");

        let git_string = String::from_utf8(git_cmd.stdout).unwrap();
        let git_re = Regex::new(r"(\w+)\s(.+)").unwrap();
        let git_caps = git_re.captures(git_string.as_str()).unwrap();

        RegressionRunInfo {
            timestamp: now,
            cpu_info: String::from("N/A"),
            git_hash: String::from(git_caps.get(1).unwrap().as_str()),
            git_log: String::from(git_caps.get(2).unwrap().as_str()),
        }
    }
}

pub fn main() {
    env_logger::init();
    info!("starting up");
    let matches = App::new("Braid regression")
        .version("0.1.0")
        .about("Braid regression test runner")
        .arg(
            Arg::with_name("config")
                .required(true)
                .value_name("FILE_IN")
                .help("Regression config Yaml file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output_dir")
                .required(true)
                .value_name("DIR_OUT")
                .help("Regression result directory")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("v")
                .short("v")
                .multiple(true)
                .help("Sets the level of verbosity"),
        )
        .get_matches();

    let run_info = RegressionRunInfo::new();

    let config: RegressionConfig = {
        let path_in_str = matches.value_of("config").unwrap();
        info!("Parsing config '{}'", path_in_str);
        let path_in = Path::new(path_in_str);
        let mut file_in = fs::File::open(&path_in).unwrap();
        let mut ser = String::new();
        file_in.read_to_string(&mut ser).unwrap();
        serde_yaml::from_str(&ser).unwrap()
    };

    let filename = format!("{}_{}.json", run_info.timestamp, config.id);
    let path_out_str = matches.value_of("output_dir").unwrap();
    let path_out = Path::new(path_out_str).join(filename.as_str());

    let seed: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut rng = XorShiftRng::from_seed(seed);

    info!("Starting tests");
    let ppc_res = match config.ppc {
        Some(ref ppc_config) => Some(run_ppc(ppc_config, &mut rng)),
        None => None,
    };

    let shapes_res = match config.shapes {
        Some(ref shapes_config) => Some(run_shapes(shapes_config, &mut rng)),
        None => None,
    };

    let geweke_res = match config.geweke {
        Some(ref geweke_config) => Some(run_geweke(geweke_config, &mut rng)),
        None => None,
    };

    let bench_res = match config.benchmark {
        Some(ref bench_config) => Some(run_benches(bench_config, &mut rng)),
        None => None,
    };

    let result = RegressionResult {
        shapes: shapes_res,
        benchmark: bench_res,
        ppc: ppc_res,
        geweke: geweke_res,
        run_info: run_info,
    };

    let mut file_out =
        fs::File::create(&path_out).expect("Failed to create output file");
    let ser = serde_json::to_string(&result).unwrap().into_bytes();
    let nbytes = file_out.write(&ser).expect("Failed to write file");

    info!("Wrote {} bytes to '{}'", nbytes, path_out.to_str().unwrap());
}
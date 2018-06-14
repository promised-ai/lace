#![feature(rustc_private)]

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate log;
extern crate env_logger;

extern crate braid;
extern crate clap;
extern crate rand;
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

use self::clap::{App, Arg};
use self::rand::prng::XorShiftRng;
use self::rand::SeedableRng;

use bench::{run_benches, BenchmarkRegressionConfig, BenchmarkResult};
use geweke::{run_geweke, GewekeRegressionConfig, GewekeRegressionResult};
use ppc::{run_ppc, PpcDistance, PpcRegressionConfig};
use shapes::{run_shapes, ShapeResult, ShapesRegressionConfig};

/// Configuration for regression testing
#[derive(Serialize, Deserialize)]
struct RegressionConfig {
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
    benchmark: Option<Vec<BenchmarkResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    geweke: Option<GewekeRegressionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ppc: Option<BTreeMap<String, Vec<PpcDistance>>>,
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
            Arg::with_name("output")
                .required(true)
                .value_name("FILE_OUT")
                .help("Regression result Yaml file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("v")
                .short("v")
                .multiple(true)
                .help("Sets the level of verbosity"),
        )
        .get_matches();

    let config: RegressionConfig = {
        let path_in_str = matches.value_of("config").unwrap();
        info!("Parsing config '{}'", path_in_str);
        let path_in = Path::new(path_in_str);
        let mut file_in = fs::File::open(&path_in).unwrap();
        let mut ser = String::new();
        file_in.read_to_string(&mut ser).unwrap();
        serde_yaml::from_str(&ser).unwrap()
    };

    let path_out_str = matches.value_of("output").unwrap();

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
    };

    info!("Writing results to '{}'", path_out_str);
    let path_out = Path::new(path_out_str);
    let mut file_out =
        fs::File::create(&path_out).expect("Failed to create output file");
    let ser = serde_json::to_string(&result).unwrap().into_bytes();
    let nbytes = file_out.write(&ser).expect("Failed to write file");

    info!("Wrote {} bytes to '{}'", nbytes, path_out_str);
}

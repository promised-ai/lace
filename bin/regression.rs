use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

use log::info;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    bench::{run_benches, BenchmarkRegressionConfig, BenchmarkResult},
    braid_opt,
    feature_error::{run_pit, FeatureErrorResult, PitRegressionConfig},
    geweke::{run_geweke, GewekeRegressionConfig, GewekeRegressionResult},
    shapes::{run_shapes, ShapeResult, ShapesRegressionConfig},
};

/// Configuration for regression testing
#[derive(Serialize, Deserialize)]
struct RegressionConfig {
    id: String,
    #[serde(default)]
    pit: Option<PitRegressionConfig>,
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
    pit: Option<BTreeMap<String, Vec<FeatureErrorResult>>>,
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
    /// Git branch name
    git_branch: String,
}

impl RegressionRunInfo {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Returns the commit hash then a space then the commit message
        let git_log_cmd = Command::new("git")
            .arg("log")
            .arg("-n 1")
            .arg("--pretty=oneline")
            .output()
            .expect("Failed to execute `git log -n 1 --pretty=oneline`");

        let git_branch_cmd = Command::new("git")
            .arg("rev-parse")
            .arg("--abbrev-ref")
            .arg("HEAD")
            .output()
            .expect("git rev-parse --abbrev-ref HEAD");

        let git_branch = String::from_utf8(git_branch_cmd.stdout).unwrap();
        let git_string = String::from_utf8(git_log_cmd.stdout).unwrap();
        let git_re = Regex::new(r"(\w+)\s(.+)").unwrap();
        let git_caps = git_re.captures(git_string.as_str()).unwrap();

        RegressionRunInfo {
            timestamp: now,
            cpu_info: String::from("N/A"),
            git_hash: String::from(git_caps.get(1).unwrap().as_str()),
            git_log: String::from(git_caps.get(2).unwrap().as_str()),
            git_branch,
        }
    }
}

pub fn regression(cmd: braid_opt::RegressionCmd) -> i32 {
    env_logger::init();

    info!("starting up");

    let run_info = RegressionRunInfo::new();

    let config: RegressionConfig = {
        info!("Parsing config '{:?}'", cmd.config);
        let path_in = Path::new(&cmd.config);
        let mut file_in = fs::File::open(&path_in).unwrap();
        let mut ser = String::new();
        file_in.read_to_string(&mut ser).unwrap();
        serde_yaml::from_str(&ser).unwrap()
    };

    let filename = match cmd.output {
        Some(s) => s,
        None => {
            let mut pathbuf = PathBuf::new();
            pathbuf.push(format!("{}_{}", run_info.timestamp, config.id));
            pathbuf.set_extension("json");
            pathbuf
        }
    };

    let mut rng = Xoshiro256Plus::seed_from_u64(19900530);

    info!("Starting tests");
    let pit_res = match config.pit {
        Some(ref pit_config) => Some(run_pit(pit_config, &mut rng)),
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
        pit: pit_res,
        geweke: geweke_res,
        run_info,
    };

    let mut file_out =
        fs::File::create(&filename).expect("Failed to create output file");
    let ser = serde_json::to_string(&result).unwrap().into_bytes();
    let nbytes = file_out.write(&ser).expect("Failed to write file");

    info!("Wrote {} bytes to '{:?}'", nbytes, filename);

    0
}

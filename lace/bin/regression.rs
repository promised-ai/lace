use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use log::info;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::bench::{run_benches, BenchmarkRegressionConfig, BenchmarkResult};
use crate::feature_error::{run_pit, FeatureErrorResult, PitRegressionConfig};
use crate::geweke::{
    run_geweke, GewekeRegressionConfig, GewekeRegressionResult,
};
use crate::opt;
use crate::shapes::{run_shapes, ShapeResult, ShapesRegressionConfig};

/// Configuration for regression testing
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegressionConfig {
    /// Name for the config. Used by the report generator to group configs and
    /// plot their results over time.
    pub id: String,
    /// Optional seed of rng
    #[serde(default)]
    pub seed: Option<u64>,
    /// If true, the regression run will save samples instead of just
    /// statistics, this will results in a much larger output file size.
    #[serde(default)]
    pub save_samples: bool,
    /// The test that measures how well lace fits against real data by using
    /// the Probability Integral Transform (PIT).
    #[serde(default)]
    pub pit: Option<PitRegressionConfig>,
    /// Tests whether the MCMC sampler is sampling from the correct posterior
    /// distribution.
    #[serde(default)]
    pub geweke: Option<GewekeRegressionConfig>,
    /// Tests how well lace fits against known zero-correlations data sets at
    /// varying scales.
    #[serde(default)]
    pub shapes: Option<ShapesRegressionConfig>,
    /// Runs timed benchmarks on states of various sizes and structure.
    #[serde(default)]
    pub benchmark: Option<BenchmarkRegressionConfig>,
}

/// Regression test results
#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct RegressionResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shapes: Option<Vec<ShapeResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<BenchmarkResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geweke: Option<GewekeRegressionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pit: Option<BTreeMap<String, Vec<FeatureErrorResult>>>,
    /// General info on the run
    pub run_info: RegressionRunInfo,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct RegressionRunInfo {
    /// Unix timestamp
    timestamp: u64,
    /// RNG seed
    seed: u64,
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
    fn new(seed: u64) -> Self {
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
            seed,
            cpu_info: String::from("N/A"), // FIXME
            git_hash: String::from(git_caps.get(1).unwrap().as_str()),
            git_log: String::from(git_caps.get(2).unwrap().as_str()),
            git_branch,
        }
    }
}

pub fn regression(cmd: opt::RegressionArgs) -> i32 {
    info!("starting up");

    let config: RegressionConfig = {
        info!("Parsing config '{:?}'", cmd.config);
        let path_in = Path::new(&cmd.config);
        let mut file_in = fs::File::open(path_in).unwrap();
        let mut ser = String::new();
        file_in.read_to_string(&mut ser).unwrap();
        serde_yaml::from_str(&ser).unwrap()
    };

    let seed = config.seed.unwrap_or_else(|| rand::thread_rng().next_u64());

    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    let run_info = RegressionRunInfo::new(seed);

    let filename = match cmd.output {
        Some(s) => s,
        None => {
            let mut pathbuf = PathBuf::new();
            pathbuf.push(format!("{}_{}", run_info.timestamp, config.id));
            pathbuf.set_extension("json");
            pathbuf
        }
    };

    info!("Starting tests");
    let save_samples = config.save_samples;
    let pit_res = config.pit.map(|pit_config| run_pit(&pit_config, &mut rng));

    let shapes_res = config.shapes.map(|shapes_config| {
        run_shapes(&shapes_config, save_samples, &mut rng)
    });

    let geweke_res = config.geweke.map(|geweke_config| {
        run_geweke(&geweke_config, save_samples, &mut rng)
    });

    let bench_res = config
        .benchmark
        .map(|bench_config| run_benches(&bench_config, &mut rng));

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

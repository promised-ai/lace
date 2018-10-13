extern crate braid;
extern crate env_logger;
extern crate rand;
extern crate regex;
extern crate serde_json;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;

use self::rand::prng::XorShiftRng;
use self::rand::SeedableRng;
use self::regex::Regex;

use bench::{run_benches, BenchmarkRegressionConfig, BenchmarkResult};
use braid_opt;
use geweke::{run_geweke, GewekeRegressionConfig, GewekeRegressionResult};
use pit::{run_pit, PitRegressionConfig, PitResult};
use shapes::{run_shapes, ShapeResult, ShapesRegressionConfig};

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
    pit: Option<BTreeMap<String, Vec<PitResult>>>,
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

pub fn regression(cmd: braid_opt::RegressionCmd) {
    env_logger::init();

    info!("starting up");

    let run_info = RegressionRunInfo::new();

    let config: RegressionConfig = {
        info!("Parsing config '{}'", cmd.config);
        let path_in = Path::new(&cmd.config);
        let mut file_in = fs::File::open(&path_in).unwrap();
        let mut ser = String::new();
        file_in.read_to_string(&mut ser).unwrap();
        serde_yaml::from_str(&ser).unwrap()
    };

    let filename = match cmd.output {
        Some(s) => s,
        None => format!("{}_{}.json", run_info.timestamp, config.id),
    };

    let path_out = Path::new(filename.as_str());

    let seed: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut rng = XorShiftRng::from_seed(seed);

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
        fs::File::create(&path_out).expect("Failed to create output file");
    let ser = serde_json::to_string(&result).unwrap().into_bytes();
    let nbytes = file_out.write(&ser).expect("Failed to write file");

    info!("Wrote {} bytes to '{}'", nbytes, path_out.to_str().unwrap());
}

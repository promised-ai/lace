extern crate braid;
extern crate regex;

use std::str::FromStr;
use structopt::StructOpt;

use self::braid::cc::transition::StateTransition;
use self::braid::cc::{ColAssignAlg, RowAssignAlg};
use self::braid::result;
use self::regex::Regex;

#[derive(Debug)]
pub struct GammaParams {
    pub a: f64,
    pub b: f64,
}

impl FromStr for GammaParams {
    type Err = result::Error;

    fn from_str(s: &str) -> result::Result<Self> {
        let re = Regex::new(r"\((\d+\.\d+),\s*(\d+\.\d+)\)").unwrap();
        match re.captures(s) {
            Some(caps) => {
                let a = f64::from_str(caps.get(1).unwrap().as_str()).unwrap();
                let b = f64::from_str(caps.get(2).unwrap().as_str()).unwrap();
                Ok(GammaParams { a, b })
            }
            None => {
                let kind = result::ErrorKind::ParseError;
                let msg = "could not parse as params tuple";
                Err(result::Error::new(kind, msg))
            }
        }
    }
}

#[derive(StructOpt, Debug)]
pub struct RegressionCmd {
    #[structopt(help = "Config YAML file")]
    pub config: String,
    #[structopt(help = "Results JSON file")]
    pub output: Option<String>,
}

#[derive(StructOpt, Debug)]
pub struct AppendCmd {
    #[structopt(short = "c", help = "Path to codebook")]
    pub codebook: Option<String>,
    #[structopt(long = "sqlite", help = "Path to SQLite3 source")]
    pub sqlite_src: Option<String>,
    #[structopt(long = "csv", help = "Path to csv source")]
    pub csv_src: Option<String>,
    pub input: String,
    pub output: String,
}

#[derive(StructOpt, Debug)]
pub struct BenchCmd {
    pub csv_src: String,
    pub codebook: String,
    #[structopt(long = "n-runs", short = "r", default_value = "1")]
    pub n_runs: usize,
    #[structopt(long = "n-iters", short = "n", default_value = "100")]
    pub n_iters: usize,
    #[structopt(long = "row-alg", default_value = "finite-cpu")]
    pub row_alg: RowAssignAlg,
    #[structopt(long = "col-alg", default_value = "finite-cpu")]
    pub col_alg: ColAssignAlg,
}

#[derive(StructOpt, Debug)]
pub struct RunCmd {
    pub output: String,
    #[structopt(short = "c", help = "Path to codebook")]
    pub codebook: Option<String>,
    #[structopt(long = "sqlite", help = "Path to SQLite3 source")]
    pub sqlite_src: Option<String>,
    #[structopt(long = "csv", help = "Path to csv source")]
    pub csv_src: Option<String>,
    #[structopt(long = "engine", help = "Path to .braid file")]
    pub engine: Option<String>,
    #[structopt(short = "t", long = "timeout", default_value = "60")]
    pub timeout: u64,
    #[structopt(long = "n-states", short = "s", long = "nstates")]
    pub nstates: usize,
    #[structopt(long = "n-iters", short = "n", default_value = "100")]
    pub n_iters: usize,
    #[structopt(long = "row-alg", default_value = "finite-cpu")]
    pub row_alg: RowAssignAlg,
    #[structopt(long = "col-alg", default_value = "finite-cpu")]
    pub col_alg: ColAssignAlg,
    #[structopt(long = "transitions")]
    pub transitions: Vec<StateTransition>,
    #[structopt(short = "o", default_value = "0")]
    pub id_offset: usize,
}

#[derive(StructOpt, Debug)]
pub struct CodebookCmd {
    pub csv_src: String,
    pub output: String,
    #[structopt(short = "g")]
    pub genomic_metadata: Option<String>,
    #[structopt(long = "alpha-params", default_value = "(1.0, 1.0)")]
    pub alpha_prior: GammaParams,
}

#[derive(StructOpt, Debug)]
#[structopt(name = "braid", about = "Expressive genetic analysis")]
pub enum BraidOpt {
    #[structopt(name = "regression", help = "Run a regression test")]
    Regression(RegressionCmd),
    #[structopt(name = "append", help = "Append columns to the table")]
    Append(AppendCmd),
    #[structopt(name = "bench", help = "Run a benchmark")]
    Bench(BenchCmd),
    #[structopt(name = "run")]
    Run(RunCmd),
    #[structopt(name = "codebook")]
    Codebook(CodebookCmd),
}

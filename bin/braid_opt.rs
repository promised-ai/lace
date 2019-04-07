extern crate braid;
extern crate regex;

use std::str::FromStr;
use structopt::StructOpt;

use braid::cc::transition::StateTransition;
use braid::cc::{ColAssignAlg, RowAssignAlg};
use braid::result;
use regex::Regex;

#[derive(Debug, PartialEq)]
pub struct GammaParams {
    pub a: f64,
    pub b: f64,
}

impl FromStr for GammaParams {
    type Err = result::Error;

    // Gamma params are going to look like this: (<shape>, <rate>), for example,
    // (1.2, 3.4).
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
    /// YAML regression configuration filename
    #[structopt(name = "YAML_IN")]
    pub config: String,
    /// JSON output filename
    #[structopt(name = "JSON_OUT")]
    pub output: Option<String>,
}

#[derive(StructOpt, Debug)]
pub struct AppendCmd {
    /// Path to the codebook
    #[structopt(short = "c")]
    pub codebook: Option<String>,
    /// Path to SQLite3 database containing new columns
    #[structopt(
        long = "sqlite",
        required_unless = "csv_src",
        conflicts_with = "csv_src"
    )]
    pub sqlite_src: Option<String>,
    /// Path to csv containing the new columns
    #[structopt(
        long = "csv",
        required_unless = "sqlite_src",
        conflicts_with = "sqlite_src"
    )]
    pub csv_src: Option<String>,
    /// .braid filename of file to append to
    pub input: String,
    /// .braid filename for output
    #[structopt(name = "BRAID_OUT")]
    pub output: String,
}

#[derive(StructOpt, Debug)]
pub struct BenchCmd {
    /// The codebook of the input data
    #[structopt(name = "CODEBOOK")]
    pub codebook: String,
    /// The path to the .csv data input
    #[structopt(name = "CSV_IN")]
    pub csv_src: String,
    /// The number of runs over which to average the benchmark
    #[structopt(long = "n-runs", short = "r", default_value = "1")]
    pub n_runs: usize,
    /// The number of iterations to run each benchmark
    #[structopt(long = "n-iters", short = "n", default_value = "100")]
    pub n_iters: usize,
    /// The row reassignment algorithm
    #[structopt(
        long = "row-alg",
        default_value = "finite_cpu",
        raw(possible_values = "&[\"finite_cpu\", \"gibbs\", \"slice\"]",)
    )]
    pub row_alg: RowAssignAlg,
    /// The column reassignment algorithm
    #[structopt(
        long = "col-alg",
        default_value = "finite_cpu",
        raw(possible_values = "&[\"finite_cpu\", \"gibbs\", \"slice\"]",)
    )]
    pub col_alg: ColAssignAlg,
}

#[derive(StructOpt, Debug)]
pub struct RunCmd {
    #[structopt(name = "BRAIDFILE_OUT")]
    pub output: String,
    /// Optinal path to codebook
    #[structopt(long = "codebook", short = "c")]
    pub codebook: Option<String>,
    /// Path to SQLite3 data soruce
    #[structopt(
        long = "sqlite",
        help = "Path to SQLite3 source",
        raw(
            required_unless_one = "&[\"engine\", \"csv_src\"]",
            conflicts_with_all = "&[\"engine\", \"csv_src\"]",
        )
    )]
    pub sqlite_src: Option<String>,
    /// Path to .csv data soruce
    #[structopt(
        long = "csv",
        help = "Path to csv source",
        raw(
            required_unless_one = "&[\"engine\", \"sqlite_src\"]",
            conflicts_with_all = "&[\"engine\", \"sqlite_src\"]",
        )
    )]
    pub csv_src: Option<String>,
    /// Path to an existing braidfile to add iterations to
    #[structopt(
        long = "engine",
        help = "Path to .braid file",
        raw(
            required_unless_one = "&[\"sqlite_src\", \"csv_src\"]",
            conflicts_with_all = "&[\"sqlite_src\", \"csv_src\"]",
        )
    )]
    pub engine: Option<String>,
    /// The maximum number of seconds to run each state. For a timeout t, the
    /// first iteration run after t seconds will be the last.
    #[structopt(short = "t", long = "timeout", default_value = "60")]
    pub timeout: u64,
    /// The number of states to create
    #[structopt(long = "n-states", short = "s", default_value = "8")]
    pub nstates: usize,
    /// The number of iterations to run each state
    #[structopt(long = "n-iters", short = "n", default_value = "100")]
    pub n_iters: usize,
    /// The row reassignment algorithm
    #[structopt(
        long = "row-alg",
        default_value = "finite_cpu",
        raw(possible_values = "&[\"finite_cpu\", \"gibbs\", \"slice\"]",)
    )]
    pub row_alg: RowAssignAlg,
    /// The column reassignment algorithm
    #[structopt(
        long = "col-alg",
        default_value = "finite_cpu",
        raw(possible_values = "&[\"finite_cpu\", \"gibbs\", \"slice\"]",)
    )]
    pub col_alg: ColAssignAlg,
    /// A list of the state transitions to run
    #[structopt(
        long = "transitions",
        use_delimiter = true,
        default_value = "column_assignment,state_alpha,row_assignment,view_alphas,feature_priors"
    )]
    pub transitions: Vec<StateTransition>,
    /// An offset for the state IDs. The n state will be named
    /// <id_offset>.state, ... , <id_offset + n - 1>.state
    #[structopt(short = "o", default_value = "0")]
    pub id_offset: usize,
}

#[derive(StructOpt, Debug)]
pub struct CodebookCmd {
    /// .csv input filename
    #[structopt(name = "CSV_IN")]
    pub csv_src: String,
    /// Codebook YAML out
    #[structopt(name = "CODEBOOK_OUT")]
    pub output: String,
    /// Optional genomic metadata
    #[structopt(short = "g")]
    pub genomic_metadata: Option<String>,
    /// Prior parameters (shape, rate) prior on CRP α
    #[structopt(long = "alpha-params", default_value = "(1.0, 1.0)")]
    pub alpha_prior: GammaParams,
}

#[derive(StructOpt, Debug)]
#[structopt(name = "braid", about = "Expressive genetic analysis")]
pub enum BraidOpt {
    /// Run a regression test
    #[structopt(name = "regression")]
    Regression(RegressionCmd),
    /// Append to columns to a braidfile
    #[structopt(name = "append")]
    Append(AppendCmd),
    /// Run a benchmark. Outputs results to stdout in YAML.
    ///
    /// EXAMPLE:
    ///
    ///     $ braid bench animals.codebook.yaml animals.csv > result.json
    #[structopt(name = "bench")]
    Bench(BenchCmd),
    /// Create and run an engine or add more iterations to an existing engine
    ///
    /// EXAMPLE - new engine from CSV:
    ///
    ///     $ braid run --csv animals.csv animals.braid
    ///
    /// EXAMPLE - add 200 iterations to an existing engine:
    ///
    ///     $ braid run --engine animals.braid --n-iters 200 animals-plus.braid
    #[structopt(name = "run")]
    Run(RunCmd),
    /// Create a default codebook from data
    ///
    /// EXAMPLE - default codebook:
    ///
    ///     $ braid codebook animals.csv animals.codebook.yaml
    ///
    /// EXAMPLE - with use-specified CRP alpha prior:
    ///
    ///     $ braid codebook --alpha-params "(2.0, 2.0)" animals.csv
    ///         animals.codebook.yaml
    #[structopt(name = "codebook")]
    Codebook(CodebookCmd),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gamma_params_valid_with_space() {
        let s = "(1.2, 2.3)";
        let g = GammaParams::from_str(s);
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), GammaParams { a: 1.2, b: 2.3 });
    }

    #[test]
    fn parse_gamma_params_valid_no_space() {
        let s = "(2.2,3.3)";
        let g = GammaParams::from_str(s);
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), GammaParams { a: 2.2, b: 3.3 });
    }

    #[test]
    fn parse_gamma_params_invalid() {
        assert!(GammaParams::from_str("(2.2,3.)").is_err());
        assert!(GammaParams::from_str("(.2,3.1)").is_err());
        assert!(GammaParams::from_str("(,3.1)").is_err());
        assert!(GammaParams::from_str("(1.2,)").is_err());
        assert!(GammaParams::from_str("(,)").is_err());
    }
}

use braid::cc::transition::StateTransition;
use braid::cc::{ColAssignAlg, RowAssignAlg};
use braid_stats::prior::CrpPrior;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
pub struct RegressionCmd {
    /// YAML regression configuration filename
    #[structopt(name = "YAML_IN")]
    pub config: PathBuf,
    /// JSON output filename
    #[structopt(name = "JSON_OUT")]
    pub output: Option<PathBuf>,
}

#[derive(StructOpt, Debug)]
pub struct AppendCmd {
    /// Path to the codebook
    #[structopt(long, short = "c", conflicts_with = "rows")]
    pub codebook: Option<PathBuf>,
    /// Path to SQLite3 database containing new columns
    #[structopt(
        long = "sqlite",
        required_unless = "csv_src",
        conflicts_with = "csv_src"
    )]
    pub sqlite_src: Option<PathBuf>,
    /// Path to csv containing the new data
    #[structopt(
        long = "csv",
        required_unless = "sqlite_src",
        conflicts_with = "sqlite_src"
    )]
    pub csv_src: Option<PathBuf>,
    /// .braid filename of file to append to
    pub input: PathBuf,
    /// .braid filename for output
    #[structopt(name = "BRAID_OUT")]
    pub output: PathBuf,
    /// Append to columns
    #[structopt(
        long = "columns",
        required_unless = "rows",
        conflicts_with = "rows"
    )]
    pub columns: bool,
    /// Append to rows
    #[structopt(
        long = "rows",
        required_unless = "columns",
        conflicts_with = "columns"
    )]
    pub rows: bool,
}

#[derive(StructOpt, Debug)]
pub struct BenchCmd {
    /// The codebook of the input data
    #[structopt(name = "CODEBOOK")]
    pub codebook: PathBuf,
    /// The path to the .csv data input
    #[structopt(name = "CSV_IN")]
    pub csv_src: PathBuf,
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
    pub output: PathBuf,
    /// Optinal path to codebook
    #[structopt(long = "codebook", short = "c")]
    pub codebook: Option<PathBuf>,
    /// Path to SQLite3 data soruce
    #[structopt(
        long = "sqlite",
        help = "Path to SQLite3 source",
        raw(
            required_unless_one = "&[\"engine\", \"csv_src\"]",
            conflicts_with_all = "&[\"engine\", \"csv_src\"]",
        )
    )]
    pub sqlite_src: Option<PathBuf>,
    /// Path to .csv data soruce
    #[structopt(
        long = "csv",
        help = "Path to csv source",
        raw(
            required_unless_one = "&[\"engine\", \"sqlite_src\"]",
            conflicts_with_all = "&[\"engine\", \"sqlite_src\"]",
        )
    )]
    pub csv_src: Option<PathBuf>,
    /// Path to an existing braidfile to add iterations to
    #[structopt(
        long = "engine",
        help = "Path to .braid file",
        raw(
            required_unless_one = "&[\"sqlite_src\", \"csv_src\"]",
            conflicts_with_all = "&[\"sqlite_src\", \"csv_src\"]",
        )
    )]
    pub engine: Option<PathBuf>,
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
    pub csv_src: PathBuf,
    /// Codebook YAML out
    #[structopt(name = "CODEBOOK_OUT")]
    pub output: PathBuf,
    /// Optional genomic metadata
    #[structopt(short = "g")]
    pub genomic_metadata: Option<PathBuf>,
    /// Prior parameters (shape, rate) prior on CRP α
    #[structopt(long = "alpha-params", default_value = "Gamma(1.0, 1.0)")]
    pub alpha_prior: CrpPrior,
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "braid",
    author = "Redpoll, LLC",
    about = "Humanistic AI engine"
)]
pub enum BraidOpt {
    /// Run a regression test
    #[structopt(name = "regression", author = "")]
    Regression(RegressionCmd),
    /// Append new rows or columns to a braidfile.
    #[structopt(name = "append", author = "")]
    Append(AppendCmd),
    /// Run a benchmark. Outputs results to stdout in YAML.
    #[structopt(name = "bench", author = "")]
    Bench(BenchCmd),
    /// Create and run an engine or add more iterations to an existing engine
    #[structopt(name = "run", author = "")]
    Run(RunCmd),
    /// Create a default codebook from data
    #[structopt(name = "codebook", author = "")]
    Codebook(CodebookCmd),
}

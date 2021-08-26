use braid::cc::config::EngineUpdateConfig;
use braid::cc::transition::StateTransition;
use braid::cc::{ColAssignAlg, RowAssignAlg};
use braid::examples::Example;
use braid_stats::prior::crp::CrpPrior;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
pub struct SummarizeCmd {
    /// The path to the braidfile to summarize
    #[structopt(name = "BRAIDFILE")]
    pub braidfile: PathBuf,
}

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
        possible_values = &["finite_cpu", "gibbs", "slice", "sams"],
    )]
    pub row_alg: RowAssignAlg,
    /// The column reassignment algorithm
    #[structopt(
        long = "col-alg",
        default_value = "finite_cpu",
        possible_values = &["finite_cpu", "gibbs", "slice"],
    )]
    pub col_alg: ColAssignAlg,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transition {
    ColumnAssignment,
    ComponentParams,
    RowAssignment,
    StateAlpha,
    ViewAlphas,
    FeaturePriors,
}

impl std::str::FromStr for Transition {
    type Err = String;

    fn from_str(s: &str) -> Result<Transition, Self::Err> {
        match s {
            "column_assignment" => Ok(Self::ColumnAssignment),
            "row_assignment" => Ok(Self::RowAssignment),
            "state_alpha" => Ok(Self::StateAlpha),
            "view_alphas" => Ok(Self::ViewAlphas),
            "feature_priors" => Ok(Self::FeaturePriors),
            "component_params" => Ok(Self::ComponentParams),
            _ => Err(format!("cannot parse '{}'", s)),
        }
    }
}

#[derive(StructOpt, Debug)]
pub struct RunCmd {
    #[structopt(name = "BRAIDFILE_OUT")]
    pub output: PathBuf,
    /// Optinal path to codebook
    #[structopt(long = "codebook", short = "c")]
    pub codebook: Option<PathBuf>,
    /// Path to .csv data soruce
    #[structopt(
        long = "csv",
        help = "Path to csv source",
        required_unless_one = &["engine"],
        conflicts_with_all = &["engine"],
    )]
    pub csv_src: Option<PathBuf>,
    /// Path to an existing braidfile to add iterations to
    #[structopt(
        long = "engine",
        help = "Path to .braid file",
        required_unless_one = &["csv-src"],
        conflicts_with_all = &["csv-src"],
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
        possible_values = &["finite_cpu", "gibbs", "slice", "sams"],
    )]
    pub row_alg: Option<RowAssignAlg>,
    /// The column reassignment algorithm
    #[structopt(
        long = "col-alg",
        possible_values = &["finite_cpu", "gibbs", "slice"],
    )]
    pub col_alg: Option<ColAssignAlg>,
    /// A list of the state transitions to run
    #[structopt(long = "transitions", use_delimiter = true)]
    pub transitions: Option<Vec<Transition>>,
    /// Path to the engine run config yaml file.
    #[structopt(
        long,
        help = "Path to Engine run config Yaml",
        conflicts_with_all = &["transitions", "col-alg", "row-alg", "timeout", "n-iters"],
    )]
    pub run_config: Option<PathBuf>,
    /// An offset for the state IDs. The n state will be named
    /// <id_offset>.state, ... , <id_offset + n - 1>.state
    #[structopt(short = "o", default_value = "0")]
    pub id_offset: usize,
    /// The PRNG seed
    #[structopt(long = "seed")]
    pub seed: Option<u64>,
    /// Initialize the engine with one view. Make sure you do not run the column
    /// assignment transition if you want to keep the columns in one view.
    #[structopt(long = "flat-columns", conflicts_with = "engine")]
    pub flat_cols: bool,
}

impl RunCmd {
    fn get_transitions(&self) -> EngineUpdateConfig {
        let row_alg = self.row_alg.unwrap_or(RowAssignAlg::FiniteCpu);
        let col_alg = self.col_alg.unwrap_or(ColAssignAlg::FiniteCpu);
        let transitions = match self.transitions {
            None => vec![
                StateTransition::ColumnAssignment(col_alg),
                StateTransition::StateAlpha,
                StateTransition::RowAssignment(row_alg),
                StateTransition::ViewAlphas,
                StateTransition::FeaturePriors,
            ],
            Some(ref ts) => ts
                .iter()
                .map(|t| match t {
                    Transition::FeaturePriors => StateTransition::FeaturePriors,
                    Transition::StateAlpha => StateTransition::StateAlpha,
                    Transition::ViewAlphas => StateTransition::ViewAlphas,
                    Transition::ComponentParams => {
                        StateTransition::ComponentParams
                    }
                    Transition::RowAssignment => {
                        StateTransition::RowAssignment(row_alg)
                    }
                    Transition::ColumnAssignment => {
                        StateTransition::ColumnAssignment(col_alg)
                    }
                })
                .collect::<Vec<_>>(),
        };

        EngineUpdateConfig {
            n_iters: self.n_iters,
            timeout: Some(self.timeout),
            transitions,
            ..Default::default()
        }
    }

    pub fn get_config(&self) -> EngineUpdateConfig {
        match self.run_config {
            Some(ref path) => {
                // TODO: proper error handling
                let f = std::fs::File::open(path.clone()).unwrap();
                serde_yaml::from_reader(f).unwrap()
            }
            None => self.get_transitions(),
        }
    }
}

#[derive(StructOpt, Debug)]
pub struct CodebookCmd {
    /// .csv input filename
    #[structopt(name = "CSV_IN")]
    pub csv_src: PathBuf,
    /// Codebook YAML out
    #[structopt(name = "CODEBOOK_OUT")]
    pub output: PathBuf,
    /// Prior parameters (shape, rate) prior on CRP α
    #[structopt(long = "alpha-params", default_value = "Gamma(1.0, 1.0)")]
    pub alpha_prior: CrpPrior,
    /// Maximum distinct values for a categorical variable
    #[structopt(short = "c", long = "category-cutoff", default_value = "20")]
    pub category_cutoff: u8,
}

#[derive(StructOpt, Debug, Clone)]
pub struct RegenExamplesCmd {
    /// The max number of iterations to run inference
    #[structopt(long, short, default_value = "1000")]
    pub n_iters: usize,
    /// The max amount of run time (sec) to run each state
    #[structopt(long, short)]
    pub timeout: Option<u64>,
    /// A list of which examples to regenerate
    #[structopt(long, min_values = 0)]
    pub examples: Option<Vec<Example>>,
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "braid",
    author = "Redpoll, LLC",
    about = "Humanistic AI engine"
)]
pub enum BraidOpt {
    /// Summarize an Engine in a braidfile
    #[structopt(name = "summarize")]
    Summarize(SummarizeCmd),
    /// Run a regression test
    #[structopt(name = "regression")]
    Regression(RegressionCmd),
    /// Run a benchmark. Outputs results to stdout in YAML.
    #[structopt(name = "bench")]
    Bench(BenchCmd),
    /// Create and run an engine or add more iterations to an existing engine
    #[structopt(name = "run")]
    Run(RunCmd),
    /// Create a default codebook from data
    #[structopt(name = "codebook")]
    Codebook(CodebookCmd),
    /// Regenerate all examples' metadata
    #[structopt(name = "regen-examples")]
    RegenExamples(RegenExamplesCmd),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn column_assignment_from_str() {
        assert_eq!(
            Transition::from_str("column_assignment").unwrap(),
            Transition::ColumnAssignment
        );
    }

    #[test]
    fn row_assignment_from_str() {
        assert_eq!(
            Transition::from_str("row_assignment").unwrap(),
            Transition::RowAssignment
        );
    }

    #[test]
    fn view_alphas_from_str() {
        assert_eq!(
            Transition::from_str("view_alphas").unwrap(),
            Transition::ViewAlphas
        );
    }

    #[test]
    fn state_alpha_from_str() {
        assert_eq!(
            Transition::from_str("state_alpha").unwrap(),
            Transition::StateAlpha
        );
    }

    #[test]
    fn feature_priors() {
        assert_eq!(
            Transition::from_str("feature_priors").unwrap(),
            Transition::FeaturePriors
        );
    }
}

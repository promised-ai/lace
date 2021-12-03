use braid::config::EngineUpdateConfig;
use braid::examples::Example;
use braid_cc::alg::{ColAssignAlg, RowAssignAlg};
use braid_cc::transition::StateTransition;
use braid_metadata::{Error, SaveConfig, SerializedType, UserInfo};
use braid_stats::prior::crp::CrpPrior;

use std::path::PathBuf;
use structopt::StructOpt;

pub(crate) trait HasUserInfo {
    fn encryption_key(&self) -> Option<&String>;
    fn profile(&self) -> Option<&String>;

    fn user_info(&self) -> Result<UserInfo, braid_metadata::Error> {
        use braid_metadata::encryption_key_from_profile;
        use braid_metadata::EncryptionKey;
        use std::convert::TryInto;

        let encryption_key = if let Some(key_string) = self.encryption_key() {
            let encryption_key: EncryptionKey = key_string.clone().into();
            let shared_key: [u8; 32] = encryption_key.try_into()?;
            Some(shared_key)
        } else if let Some(profile) = self.profile() {
            encryption_key_from_profile(profile)?
        } else {
            None
        };

        Ok(UserInfo {
            encryption_key,
            profile: self.profile().cloned(),
        })
    }
}

#[derive(StructOpt, Debug)]
pub struct SummarizeArgs {
    /// The path to the braidfile to summarize
    #[structopt(name = "BRAIDFILE")]
    pub braidfile: PathBuf,
    /// Encryption key for working with encrypted engines
    #[structopt(
        short = "k",
        long = "encryption-key",
        conflicts_with = "profile"
    )]
    pub encryption_key: Option<String>,
    /// Profile to use for looking up encryption keys, etc
    #[structopt(
        short = "p",
        long = "profile",
        conflicts_with = "encryption-key"
    )]
    pub profile: Option<String>,
}

impl HasUserInfo for SummarizeArgs {
    fn encryption_key(&self) -> Option<&String> {
        self.encryption_key.as_ref()
    }
    fn profile(&self) -> Option<&String> {
        self.profile.as_ref()
    }
}

#[derive(StructOpt, Debug)]
pub struct RegressionArgs {
    /// YAML regression configuration filename
    #[structopt(name = "YAML_IN")]
    pub config: PathBuf,
    /// JSON output filename
    #[structopt(name = "JSON_OUT")]
    pub output: Option<PathBuf>,
}

#[derive(StructOpt, Debug)]
pub struct BenchArgs {
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
pub struct RunArgs {
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
    #[structopt(short = "t", long = "timeout")]
    pub timeout: Option<u64>,
    /// The number of states to create
    #[structopt(long = "n-states", short = "s", default_value = "8")]
    pub nstates: usize,
    /// The number of iterations to run each state
    #[structopt(long = "n-iters", short = "n", required_unless = "run-config")]
    pub n_iters: Option<usize>,
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
    /// Format to save the output
    #[structopt(short = "f", long = "output-format")]
    pub output_format: Option<SerializedType>,
    /// Encryption key for working with encrypted engines
    #[structopt(
        short = "k",
        long = "encryption-key",
        conflicts_with = "profile"
    )]
    pub encryption_key: Option<String>,
    /// Profile to use for looking up encryption keys, etc
    #[structopt(long = "profile", conflicts_with = "encryption-key")]
    pub profile: Option<String>,
}

impl RunArgs {
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
            n_iters: self.n_iters.unwrap(),
            timeout: self.timeout,
            transitions,
            ..Default::default()
        }
    }

    pub fn engine_update_config(&self) -> EngineUpdateConfig {
        match self.run_config {
            Some(ref path) => {
                // TODO: proper error handling
                let f = std::fs::File::open(path.clone()).unwrap();
                serde_yaml::from_reader(f).unwrap()
            }
            None => self.get_transitions(),
        }
    }

    pub fn save_config(&self) -> Result<SaveConfig, Error> {
        let user_info = self.user_info()?;

        let output_format =
            match (self.output_format, self.encryption_key.is_some()) {
                (Some(fmt), _) => fmt,
                (None, true) => SerializedType::Encrypted,
                (None, false) => SerializedType::Bincode,
            };

        Ok(SaveConfig {
            metadata_version: braid_metadata::latest::METADATA_VERSION,
            serialized_type: output_format,
            user_info,
        })
    }
}

impl HasUserInfo for RunArgs {
    fn encryption_key(&self) -> Option<&String> {
        self.encryption_key.as_ref()
    }
    fn profile(&self) -> Option<&String> {
        self.profile.as_ref()
    }
}

#[derive(StructOpt, Debug)]
pub struct CodebookArgs {
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
    /// Skip running sanity checks on input data such as proportion of missing
    /// values
    #[structopt(long)]
    pub no_checks: bool,
}

#[derive(StructOpt, Debug, Clone)]
pub struct RegenExamplesArgs {
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
pub enum Opt {
    /// Summarize an Engine in a braidfile
    #[structopt(name = "summarize")]
    Summarize(SummarizeArgs),
    /// Run a regression test
    #[structopt(name = "regression")]
    Regression(RegressionArgs),
    /// Run a benchmark. Outputs results to stdout in YAML.
    #[structopt(name = "bench")]
    Bench(BenchArgs),
    /// Create and run an engine or add more iterations to an existing engine
    #[structopt(name = "run")]
    Run(RunArgs),
    /// Create a default codebook from data
    #[structopt(name = "codebook")]
    Codebook(CodebookArgs),
    /// Regenerate all examples' metadata
    #[structopt(name = "regen-examples")]
    RegenExamples(RegenExamplesArgs),
    /// Generate an encryption key
    #[structopt(name = "keygen")]
    GenerateEncyrptionKey,
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

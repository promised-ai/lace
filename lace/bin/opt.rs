use clap::Parser;
use lace::cc::alg::{ColAssignAlg, RowAssignAlg};
use lace::cc::transition::StateTransition;
use lace::config::EngineUpdateConfig;
use lace::data::DataSource;
use lace::examples::Example;
use lace::stats::rv::dist::Gamma;
use lace_metadata::{Error, FileConfig, SerializedType};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug)]
pub struct SummarizeArgs {
    /// The path to the lacefile to summarize
    #[clap(name = "LACEFILE")]
    pub lacefile: PathBuf,
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
            _ => Err(format!("cannot parse '{s}'")),
        }
    }
}

// TODO: when clap 4.0 comes out, we might be able to smoosh all the input types
// into an enum-derived ArgGroup.
#[derive(Parser, Debug)]
pub struct RunArgs {
    #[clap(name = "LACEFILE_OUT")]
    pub output: PathBuf,
    /// Optinal path to codebook
    #[clap(long = "codebook", short = 'c')]
    pub codebook: Option<PathBuf>,
    /// Path to .csv data source. May be compressed.
    #[clap(long = "csv", group = "input")]
    pub csv_src: Option<PathBuf>,
    /// Path to Apache IPC (feather v2) data source
    #[clap(long = "ipc", group = "input")]
    pub ipc_src: Option<PathBuf>,
    /// Path to parquet data source
    #[clap(long = "parquet", group = "input")]
    pub parquet_src: Option<PathBuf>,
    /// Path to .json or .jsonl data source. Note that if the extension does not
    /// match, lace will assume the data are in JSON line format
    #[clap(long = "json", group = "input")]
    pub json_src: Option<PathBuf>,
    /// Path to an existing lacefile to add iterations to
    #[clap(long = "engine", help = "Path to .lace file", group = "input")]
    pub engine: Option<PathBuf>,
    /// The maximum number of seconds to run each state. For a timeout t, the
    /// first iteration run after t seconds will be the last.
    #[clap(short = 't', long = "timeout")]
    pub timeout: Option<u64>,
    /// The number of states to create
    #[clap(long = "n-states", short = 's', default_value = "8")]
    pub nstates: usize,
    /// The number of iterations to run each state
    #[clap(long = "n-iters", short = 'n', required_unless = "run-config")]
    pub n_iters: Option<usize>,
    /// The number of iterations between state saves
    #[clap(short = 'C', long, conflicts_with = "run-config")]
    pub checkpoint: Option<usize>,
    /// The row reassignment algorithm
    #[clap(
        long = "row-alg",
        possible_values = &["finite_cpu", "gibbs", "slice", "sams"],
    )]
    pub row_alg: Option<RowAssignAlg>,
    /// The column reassignment algorithm
    #[clap(
        long = "col-alg",
        possible_values = &["finite_cpu", "gibbs", "slice"],
    )]
    pub col_alg: Option<ColAssignAlg>,
    /// A list of the state transitions to run
    #[clap(long = "transitions", use_delimiter = true)]
    pub transitions: Option<Vec<Transition>>,
    /// Path to the engine run config yaml file.
    #[clap(
        long,
        help = "Path to Engine run config Yaml",
        conflicts_with_all = &["transitions", "col-alg", "row-alg", "n-iters"],
    )]
    pub run_config: Option<PathBuf>,
    /// An offset for the state IDs. The n state will be named
    /// <id_offset>.state, ... , <id_offset + n - 1>.state
    #[clap(short = 'o', default_value = "0")]
    pub id_offset: usize,
    /// The PRNG seed
    #[clap(long = "seed")]
    pub seed: Option<u64>,
    /// Initialize the engine with one view. Make sure you do not run the column
    /// assignment transition if you want to keep the columns in one view.
    #[clap(short = 'F', long = "flat-columns", conflicts_with = "engine")]
    pub flat_cols: bool,
    /// Do not run the column reassignment kernel
    #[clap(short = 'R', long = "no-column-reassign")]
    pub no_col_reassign: bool,
    /// Format to save the output
    #[clap(short = 'f', long = "output-format")]
    pub output_format: Option<SerializedType>,
    /// Do not display run progress
    #[clap(long, short)]
    pub quiet: bool,
}

fn filter_transitions(
    mut transitions: Vec<StateTransition>,
    no_col_reassign: bool,
) -> Vec<StateTransition> {
    transitions
        .drain(..)
        .filter(|t| {
            !(no_col_reassign
                && matches!(t, StateTransition::ColumnAssignment(_)))
        })
        .collect()
}

impl RunArgs {
    fn get_transitions(&self) -> EngineUpdateConfig {
        let row_alg = self.row_alg.as_ref().unwrap_or(&RowAssignAlg::Slice);
        let col_alg = self.col_alg.unwrap_or(ColAssignAlg::Slice);
        let transitions = match self.transitions {
            None => vec![
                StateTransition::ColumnAssignment(col_alg),
                StateTransition::StateAlpha,
                StateTransition::RowAssignment(row_alg.clone()),
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
                        StateTransition::RowAssignment(row_alg.clone())
                    }
                    Transition::ColumnAssignment => {
                        StateTransition::ColumnAssignment(col_alg)
                    }
                })
                .collect::<Vec<_>>(),
        };

        EngineUpdateConfig {
            n_iters: self.n_iters.unwrap(),
            transitions: filter_transitions(transitions, self.no_col_reassign),
            ..Default::default()
        }
    }

    pub fn engine_update_config(&self) -> EngineUpdateConfig {
        match self.run_config {
            Some(ref path) => {
                // TODO: proper error handling
                let f = std::fs::File::open(path.clone()).unwrap();
                let mut config: EngineUpdateConfig =
                    serde_yaml::from_reader(f).unwrap();
                config.transitions = filter_transitions(
                    config.transitions,
                    self.no_col_reassign,
                );
                config
            }
            None => self.get_transitions(),
        }
    }

    pub fn file_config(&self) -> Result<FileConfig, Error> {
        Ok(FileConfig {
            metadata_version: lace_metadata::latest::METADATA_VERSION,
            serialized_type: self.output_format.unwrap_or_default(),
        })
    }

    #[allow(clippy::manual_map)]
    pub fn data_source(&self) -> Option<DataSource> {
        if let Some(ref path) = self.csv_src {
            Some(DataSource::Csv(path.clone()))
        } else if let Some(ref path) = self.parquet_src {
            Some(DataSource::Parquet(path.clone()))
        } else if let Some(ref path) = self.ipc_src {
            Some(DataSource::Ipc(path.clone()))
        } else if let Some(ref path) = self.json_src {
            Some(DataSource::Json(path.clone()))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct GammaParameters {
    pub shape: f64,
    pub rate: f64,
}

fn params_parse_fail(s: &str) -> String {
    format!("`{s}` cannot be converted to GammaParameters")
}

impl Default for GammaParameters {
    fn default() -> Self {
        GammaParameters {
            shape: 1.0,
            rate: 1.0,
        }
    }
}

impl FromStr for GammaParameters {
    type Err = String;

    // Params are of the form "shape, rate", e.g. "1.0, 2.0"
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.trim().split(',').collect();
        if parts.len() != 2 {
            return Err(params_parse_fail(s));
        }
        let shape: f64 =
            parts[0].trim().parse().map_err(|_| params_parse_fail(s))?;
        let rate: f64 =
            parts[1].trim().parse().map_err(|_| params_parse_fail(s))?;

        Ok(GammaParameters { shape, rate })
    }
}

impl TryInto<Gamma> for GammaParameters {
    type Error = lace::stats::rv::dist::GammaError;
    fn try_into(self) -> Result<Gamma, Self::Error> {
        Gamma::new(self.shape, self.rate)
    }
}

#[derive(Parser, Debug)]
pub struct CodebookArgs {
    /// .csv input filename
    #[clap(long = "csv", group = "src")]
    pub csv_src: Option<PathBuf>,
    /// .json or .jsonl input filename
    #[clap(long = "json", group = "src")]
    pub json_src: Option<PathBuf>,
    /// Apache IPC (Feather v2) input filename
    #[clap(long = "ipc", group = "src")]
    pub ipc_src: Option<PathBuf>,
    /// Parquet input filename
    #[clap(long = "parquet", group = "src")]
    pub parquet_src: Option<PathBuf>,
    /// CRP alpha prior on columns and rows
    #[clap(long, default_value = "1.0, 1.0")]
    pub alpha_prior: GammaParameters,
    /// Codebook out. May be either json or yaml
    #[clap(name = "CODEBOOK_OUT")]
    pub output: PathBuf,
    /// Maximum distinct values for a categorical variable
    #[clap(short = 'c', long = "category-cutoff", default_value = "20")]
    pub category_cutoff: u8,
    /// Skip running sanity checks on input data such as proportion of missing
    /// values
    #[clap(long)]
    pub no_checks: bool,
    /// Do not use hyper prior inference. Instead, use empirical priors derived
    /// from the data.
    #[clap(long)]
    pub no_hyper: bool,
}

#[derive(Parser, Debug, Clone)]
pub struct RegenExamplesArgs {
    /// The max number of iterations to run inference
    #[clap(long, short, default_value = "5000")]
    pub n_iters: usize,
    /// The max amount of run time (sec) to run each state
    #[clap(long, short)]
    pub timeout: Option<u64>,
    /// A list of which examples to regenerate
    #[clap(long, min_values = 0)]
    pub examples: Option<Vec<Example>>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Parser, Debug)]
#[clap(
    name = "lace",
    author = "Promised AI",
    about = "Humanistic AI engine",
    version
)]
pub enum Opt {
    /// Summarize an Engine in a lace model
    #[clap(name = "summarize")]
    Summarize(SummarizeArgs),
    /// Create and run an engine or add more iterations to an existing engine
    #[clap(name = "run")]
    Run(RunArgs),
    /// Create a default codebook from data. You may save the output as yaml or json.
    #[clap(name = "codebook")]
    Codebook(CodebookArgs),
    /// Regenerate all examples' metadata
    #[clap(name = "regen-examples")]
    RegenExamples(RegenExamplesArgs),
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

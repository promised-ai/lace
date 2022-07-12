use std::collections::{BTreeMap, HashMap};
use std::env;

use crate::api::obj::{
    Datum, FType, Given, ImputeUncertaintyType, MiType, PredictUncertaintyType,
    Row, StateDiagnostics, StateTransition, SummaryStatistics, WriteMode,
};
use braid::codebook::{Codebook, ColMetadata, ColMetadataList};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use utoipa::Component;

use crate::api::TooLong;

/// Limits on the request computations to prevent users from locking up the
/// server. The values are set at rutime with the following environmental
/// variables:
///
/// - BRAID_MAX_DEPPROB_QUERIES
/// - BRAID_MAX_ROWSIM_QUERIES
/// - BRAID_MAX_MI_EVALS
/// - BRAID_MAX_SIMULATE_EVALS
/// - BRAID_MAX_LOGP_EVALS
/// - BRAID_MAX_GET_DATA_QUERIES
/// - BRAID_MAX_SUMMARY_QUERIES
///
/// If a variable is not set, or does not parse, a default will be used. Read
/// the source for defaults.
#[derive(
    Serialize,
    Deserialize,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Clone,
    Debug,
    Component,
)]
#[serde(deny_unknown_fields)]
pub struct RequestLimits {
    /// The maximum allowed dependence probability column pairs allowed in one
    /// query
    pub max_depprob_queries: usize,
    /// The maximum allowed row similarity row pairs allowed in one query
    pub max_rowsim_queries: usize,
    /// The maximum number of mutual information evaluations (defined as a
    /// sample) allowed in one query. For example, if the `max_mi_evals` is
    /// 100,000 and the requested number of samples per mi computations is set
    /// to 1,000, the user will be allowed 100 column pairs per query.
    pub max_mi_evals: usize,
    /// The maximum number of cells allowed to be sampled per query, which is
    /// the number of dimensions multiplied by the number of simulations.
    pub max_simulate_evals: usize,
    /// The maximum number of cells allowed to be computed per query, which is
    /// the number of dimensions multiplied by the number of rows.
    pub max_logp_evals: usize,
    /// The maximum number of indices the user can request data for in a query
    pub max_get_data_queries: usize,
    /// The maximum number of columns the user can request summaries for
    pub max_summary_queries: usize,
}

fn try_var(varname: &str) -> Option<usize> {
    env::var(varname).ok().and_then(|s| s.parse().ok())
}

lazy_static! {
    static ref REQUEST_LIMITS: RequestLimits = {
        RequestLimits {
            max_depprob_queries: try_var("BRAID_MAX_DEPPROB_QUERIES")
                .unwrap_or(5_000),
            max_rowsim_queries: try_var("BRAID_MAX_ROWSIM_QUERIES")
                .unwrap_or(5_000),
            max_mi_evals: try_var("BRAID_MAX_MI_EVALS").unwrap_or(100_000),
            max_simulate_evals: try_var("BRAID_MAX_SIMULATE_EVALS")
                .unwrap_or(1_000_000),
            max_logp_evals: try_var("BRAID_MAX_LOGP_EVALS").unwrap_or(100_000),
            max_get_data_queries: try_var("BRAID_MAX_GET_DATA_QUERIES")
                .unwrap_or(100_000),
            max_summary_queries: try_var("BRAID_MAX_SUMMARY_QUERIES")
                .unwrap_or(100_000),
        }
    };
}

macro_rules! never_too_long {
    ($kind: ty, $name: expr) => {
        impl TooLong for $kind {
            fn too_long(&self) -> bool {
                false
            }

            fn too_long_msg(&self) -> String {
                format!(
                    "This query was reported as being too long, \
                     but `{}` queries should never be too long. Please report \
                     this issue to the developers",
                    $name
                )
            }
        }
    };
}

pub fn request_limits() -> RequestLimits {
    REQUEST_LIMITS.clone()
}

/// The response from the `/version` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct VersionResponse {
    pub version: String,
}

/// The response from the `/request_limits` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RequestLimitsResponse {
    pub limits: RequestLimits,
}

/// The response to the `/diagnostics` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct StateDiagnosticsResponse {
    /// `diagnostics[i]` corresponds to the diagnostics for the i<sup>th</sup>
    /// state in the `Oracle`
    pub diagnostics: Vec<StateDiagnostics>,
}

/// The response from the `/nstates` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct NStatesResponse {
    /// The number of states in the Oracle
    pub nstates: usize,
}

/// The response from the `/shape` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ShapeResponse {
    /// The number of rows in the table
    pub n_rows: usize,
    /// The number of columns in the table
    pub n_cols: usize,
}

/// The response to the `/codebook` route
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct CodebookResponse {
    /// The `Oracle`'s codebook
    pub codebook: Codebook,
}

/// The response to the `/ftype` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct FTypeResponse {
    pub ftype: FType,
}

/// The response to the `/ftypes` route
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct FTypesResponse {
    /// A an ordered list of the `FTypes` of each feature/column. `ftypes[i]`
    /// corresponds to the type of the i<sup>th</sup> feature/column.
    pub ftypes: Vec<FType>,
}

/// Get data request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SummarizeColumnsRequest {
    /// A vector of column indices
    pub col_ixs: Vec<usize>,
}

impl TooLong for SummarizeColumnsRequest {
    fn too_long(&self) -> bool {
        self.col_ixs.len() > REQUEST_LIMITS.max_summary_queries
    }

    fn too_long_msg(&self) -> String {
        format!(
            "The summary query is limited to {} columns per request",
            REQUEST_LIMITS.max_summary_queries
        )
    }
}

/// Get data response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SummarizeColumnsResponse {
    /// A vector of `(row_ix, col_ix, value)` tuples
    pub summaries: BTreeMap<usize, SummaryStatistics>,
}

/// The request for dependence probability
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
//#[component(example = json!({"col_pairs": [(0, 1), (1, 2)]}))]
pub struct DepprobRequest {
    /// A vector of pairs of column indices
    pub col_pairs: Vec<(usize, usize)>,
}

impl TooLong for DepprobRequest {
    fn too_long(&self) -> bool {
        self.col_pairs.len() > REQUEST_LIMITS.max_depprob_queries
    }

    fn too_long_msg(&self) -> String {
        format!(
            "depprob queries may have at most {} col_pairs",
            REQUEST_LIMITS.max_depprob_queries
        )
    }
}

/// The dependence probability response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DepprobResponse {
    /// A vector of tuples `(col_a, col_b, depprob(col_a, col_b))`
    pub depprob: Vec<(usize, usize, f64)>,
}

/// Row similarity request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RowsimRequest {
    /// A vector of row index pairs
    pub row_pairs: Vec<(usize, usize)>,
    /// A vector of option column indices through which to constrain the
    /// similarity
    #[serde(default)]
    pub wrt: Vec<usize>,
    /// If `true` weight row similarity by number of columns instead of number
    /// of views
    #[serde(default)]
    pub col_weighted: bool,
}

impl TooLong for RowsimRequest {
    fn too_long(&self) -> bool {
        self.row_pairs.len() > REQUEST_LIMITS.max_rowsim_queries
    }

    fn too_long_msg(&self) -> String {
        format!(
            "rowsim queries may have at most {} row_pairs",
            REQUEST_LIMITS.max_rowsim_queries
        )
    }
}

/// Row similarity response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RowsimResponse {
    /// A vector of `(row_a, row_b, rowsim(row_a, row_b))`
    pub rowsim: Vec<(usize, usize, f64)>,
}

/// Novelty request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct NoveltyRequest {
    /// The row indices
    pub row_ixs: Vec<usize>,
    /// Optional indices of columns to add context
    #[serde(default)]
    pub wrt: Vec<usize>,
}

never_too_long!(NoveltyRequest, "novelty");

/// Novelty response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct NoveltyResponse {
    /// List containing (row index, its novelty)
    pub novelty: Vec<(usize, f64)>,
}

/// Mutual information request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MiRequest {
    /// A vector of column index pair tuples
    pub col_pairs: Vec<(usize, usize)>,
    /// The number of samples to use to estimate the mutual information
    pub n: usize,
    /// The type of mutual information to compute
    pub mi_type: MiType,
}

impl TooLong for MiRequest {
    fn too_long(&self) -> bool {
        let tot = self.n * self.col_pairs.len();
        tot > REQUEST_LIMITS.max_mi_evals
    }

    fn too_long_msg(&self) -> String {
        format!(
            "The number of samples, `n`, multiplied by the number of \
             column pairs, `col_pairs`, may be at most {}. If you would like \
             estimates based on higher `n`, you may average multiple queries.",
            REQUEST_LIMITS.max_mi_evals,
        )
    }
}

/// Mutual information response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MiResponse {
    // TODO: Also return relative sample size
    /// A vector of `(col_a, col_b, mi(col_a, col_b)` tuples
    pub mi: Vec<(usize, usize, f64)>,
}

/// Joint entropy request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct EntropyRequest {
    /// The target column indices
    pub col_ixs: Vec<usize>,
    /// The number of samples for QMC
    pub n: usize,
}

never_too_long!(EntropyRequest, "entropy");

/// Joint response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct EntropyResponse {
    /// The joint entropy
    pub entropy: f64,
}

/// Information proportion request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct InfoPropRequest {
    /// The target column indices
    pub target_ixs: Vec<usize>,
    /// The predictor column indices
    pub predictor_ixs: Vec<usize>,
    /// The number of samples for QMC
    pub n: usize,
}

never_too_long!(InfoPropRequest, "info_prop");

/// Information proportion response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct InfoPropResponse {
    /// The information proportion
    pub info_prop: f64,
}

/// Surprisal request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SurprisalRequest {
    /// The column index
    pub col_ix: usize,
    /// The row index
    pub row_ixs: Vec<usize>,
    /// Specify the target datum
    #[serde(default)]
    pub target_data: Option<Vec<Datum>>,
    /// The states over which to compute surprisal
    #[serde(default)]
    pub state_ixs: Option<Vec<usize>>,
}

never_too_long!(SurprisalRequest, "surprisal");

/// Surprisal response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SurprisalResponse {
    /// The column index
    pub col_ix: usize,
    /// The row index
    pub row_ixs: Vec<usize>,
    /// The value in the target cell
    pub values: Vec<Datum>,
    /// The surprisal of `value` in the cell. Is `None` if `value` is `Missing`.
    pub surprisal: Vec<Option<f64>>,
}

/// Get data request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct GetDataRequest {
    /// A vector of `(row_ix, col_ix)` tuples
    pub ixs: Vec<(usize, usize)>,
}

impl TooLong for GetDataRequest {
    fn too_long(&self) -> bool {
        self.ixs.len() > REQUEST_LIMITS.max_get_data_queries
    }

    fn too_long_msg(&self) -> String {
        format!(
            "The get_data query is limited to {} indices per request",
            REQUEST_LIMITS.max_get_data_queries
        )
    }
}

/// Get data response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct GetDataResponse {
    /// A vector of `(row_ix, col_ix, value)` tuples
    pub values: Vec<(usize, usize, Datum)>,
}

/// Log PMF/PDF request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct LogpRequest {
    /// An NxD vector of vector containing he values for which to compute the
    /// log PMF/PDF. Rows correspond to grouped values of a joint probability.
    pub values: Vec<Vec<Datum>>,
    /// The column indices of the values
    pub col_ixs: Vec<usize>,
    /// A list of `(col_ix, value)` tuples
    #[serde(default)]
    pub given: Given,
    /// Optional indices of the state to use for computations
    #[serde(default)]
    pub state_ixs: Option<Vec<usize>>,
}

impl TooLong for LogpRequest {
    fn too_long(&self) -> bool {
        let nvals = self.values.len();
        let ncols = self.col_ixs.len();
        (nvals * ncols) > REQUEST_LIMITS.max_logp_evals
    }

    fn too_long_msg(&self) -> String {
        format!(
            "The number of values multiplied by the number of columns per \
             value, must be less than {}. You may break your query up by \
             values. If you wish to compute `logp` for more than {} columns per \
             value please let the developers know.",
            REQUEST_LIMITS.max_logp_evals, REQUEST_LIMITS.max_logp_evals,
        )
    }
}

/// Log PMF/PDF response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct LogpResponse {
    /// A vector of log PDFs/PMFs corresponding to each row (outer vec) of the
    /// request `values`
    pub logp: Vec<f64>,
}

/// Scaled log PMF/PDF request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct LogpScaledRequest {
    /// An NxD vector of vector containing he values for which to compute the
    /// log PMF/PDF. Rows correspond to grouped values of a joint probability.
    pub values: Vec<Vec<Datum>>,
    /// The column indices of the values
    pub col_ixs: Vec<usize>,
    /// A list of `(col_ix, value)` tuples
    #[serde(default)]
    pub given: Given,
    /// Optional indices of the state to use for computations
    #[serde(default)]
    pub state_ixs: Option<Vec<usize>>,
}

impl TooLong for LogpScaledRequest {
    fn too_long(&self) -> bool {
        let nvals = self.values.len();
        let ncols = self.col_ixs.len();
        (nvals * ncols) > REQUEST_LIMITS.max_logp_evals
    }

    fn too_long_msg(&self) -> String {
        format!(
            "The number of values multiplied by the number of columns per \
             value, must be less than {}. You may break your query up by \
             values. If you wish to compute `logp` for more than {} columns per \
             value please let the developers know.",
            REQUEST_LIMITS.max_logp_evals, REQUEST_LIMITS.max_logp_evals,
        )
    }
}

/// Scaled log PMF/PDF response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct LogpScaledResponse {
    /// A vector of scaled log PDFs/PMFs corresponding to each row (outer vec)
    /// of the request `values`
    pub logp: Vec<f64>,
}

/// Draw request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DrawRequest {
    /// The column index
    pub col_ix: usize,
    /// The row index
    pub row_ix: usize,
    /// The number of draws to take
    pub n: usize,
    /// Optional RNG seed
    #[serde(default)]
    pub seed: Option<u64>,
}

never_too_long!(DrawRequest, "draw");

/// Draw response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DrawResponse {
    /// An n-length vector of draws from the cell
    pub values: Vec<Datum>,
}

/// Simulate request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SimulateRequest {
    /// The column indeices to simulate
    pub col_ixs: Vec<usize>,
    /// A list of `(col_ix, value)` condition tuples
    #[serde(default)]
    pub given: Given,
    /// The number of draws/simulations
    pub n: usize,
    /// The states from which to simulate. If None, use all.
    #[serde(default)]
    states_ixs: Option<Vec<usize>>,
    /// Optional RNG seed
    #[serde(default)]
    pub seed: Option<u64>,
}

impl TooLong for SimulateRequest {
    fn too_long(&self) -> bool {
        if self.n == 1 {
            false
        } else {
            (self.col_ixs.len() * self.n) > REQUEST_LIMITS.max_simulate_evals
        }
    }
    fn too_long_msg(&self) -> String {
        format!(
            "The number of simulations, `n`, multiplied by the number of \
             columns to simulate, must be less than {}. To achieve higher `n`, \
             perform multiple queries with. If you wish to simulate more than {} \
             columns simultaneously, please let the developers know.",
            REQUEST_LIMITS.max_simulate_evals, REQUEST_LIMITS.max_simulate_evals,
        )
    }
}

/// Simulation response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SimulateResponse {
    /// `values[i][j]` is the i<sup>th</sup> draw of the j<sup>th<sup> column in
    /// the request
    pub values: Vec<Vec<Datum>>,
}

/// Impute request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ImputeRequest {
    /// The column index
    pub row_ix: usize,
    /// The row index
    pub col_ix: usize,
    /// Which uncertainty type to use. If None, uncertainty is not computed
    #[serde(default)]
    pub uncertainty_type: Option<ImputeUncertaintyType>,
}

never_too_long!(ImputeRequest, "impute");

/// Impute response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ImputeResponse {
    /// The row index
    pub row_ix: usize,
    /// The column index
    pub col_ix: usize,
    /// The imputed value
    pub value: Datum,
    /// The uncertainty
    pub uncertainty: Option<f64>,
}

/// Predict request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PredictRequest {
    /// The column/feature to predict
    pub col_ix: usize,
    /// The a vector of `(col_ix, value)` tuples conditioning the prediction,
    /// e.g. argmax[ p(col_x | given) ]
    #[serde(default)]
    pub given: Given,
    /// Which uncertainty type to use. If None, uncertainty is not computed
    #[serde(default)]
    pub uncertainty_type: Option<PredictUncertaintyType>,
}

never_too_long!(PredictRequest, "predict");

/// Predict response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PredictResponse {
    /// The predicted value
    pub value: Datum,
    /// The impute uncertainty
    pub uncertainty: Option<f64>,
}

/// Feature error response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct FeatureErrorResponse {
    /// The cumulative error
    pub error: f64,
    /// The centroid of the error
    pub centroid: f64,
}

/// Insert/update data request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct InsertDataRequest {
    /// The data rows
    pub rows: Vec<Row>,
    /// The metadata for any new columns to be inserted
    #[serde(default)]
    pub new_col_metadata: Option<ColMetadataList>,
    #[serde(default)]
    pub suppl_metadata: Option<HashMap<String, ColMetadata>>,
    /// The optional write mode. Defaults to only overwrite missing cells.
    #[serde(default)]
    pub write_mode: Option<WriteMode>,
}

never_too_long!(InsertDataRequest, "insert_data");

/// Insert/update data response showing the work done
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct InsertDataResponse {
    /// The number of new rows added
    pub new_rows: usize,
    /// The number of new columns added
    pub new_cols: usize,
}

/// Update engine request
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct UpdateRequest {
    pub n_iters: usize,
    /// Timeout in seconds.
    #[serde(default)]
    pub timeout: Option<u64>,
    /// Which transitions to run
    pub transitions: Vec<StateTransition>,
}

impl From<UpdateRequest> for braid::EngineUpdateConfig {
    fn from(mut req: UpdateRequest) -> Self {
        Self {
            n_iters: req.n_iters,
            timeout: req.timeout,
            checkpoint: None,
            transitions: req
                .transitions
                .drain(..)
                .map(braid::StateTransition::from)
                .collect(),
            save_config: None,
        }
    }
}

never_too_long!(UpdateRequest, "update");

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AssignmentResponse {
    /// The view indices to which each column is assigned
    pub column_assignment: Vec<usize>,
    /// For each view, the indices to which each row is assigned
    pub row_assignments: Vec<Vec<usize>>,
}

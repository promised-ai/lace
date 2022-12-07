// use braid::codebook::{Codebook, ColMetadata, ColMetadataList};
// use braid::{
//     Datum, FType, Given, ImputeUncertaintyType, MiType, PredictUncertaintyType,
//     Row, StateDiagnostics, StateTransition, SummaryStatistics, WriteMode,
// };
use serde::{Deserialize, Serialize};
use utoipa::Component;

#[derive(Serialize, Deserialize, Component, Clone, Debug, PartialEq)]
#[serde(rename = "datum")]
pub enum Datum {
    #[serde(rename = "continuous")]
    Continuous(f64),
    #[serde(rename = "categorical")]
    Categorical(u8),
    #[serde(rename = "count")]
    Count(u32),
    #[serde(rename = "missing")]
    Missing,
}

impl Datum {
    pub fn is_continuous(&self) -> bool {
        matches!(self, Self::Continuous(_))
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self, Self::Categorical(_))
    }

    pub fn is_count(&self) -> bool {
        matches!(self, Self::Count(_))
    }
}

impl From<Datum> for braid::Datum {
    fn from(x: Datum) -> Self {
        match x {
            Datum::Continuous(y) => Self::Continuous(y),
            Datum::Categorical(y) => Self::Categorical(y),
            Datum::Count(y) => Self::Count(y),
            Datum::Missing => Self::Missing,
        }
    }
}

impl From<braid::Datum> for Datum {
    fn from(x: braid::Datum) -> Self {
        match x {
            braid::Datum::Continuous(y) => Self::Continuous(y),
            braid::Datum::Categorical(y) => Self::Categorical(y),
            braid::Datum::Count(y) => Self::Count(y),
            braid::Datum::Missing => Self::Missing,
            _ => panic!("Label not supported"),
        }
    }
}

/// Feature type
#[derive(Serialize, Deserialize, Component, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FType {
    Continuous,
    Categorical,
    Count,
}

impl From<FType> for braid::FType {
    fn from(ftype: FType) -> Self {
        match ftype {
            FType::Continuous => Self::Continuous,
            FType::Categorical => Self::Categorical,
            FType::Count => Self::Count,
        }
    }
}

impl From<braid::FType> for FType {
    fn from(ftype: braid::FType) -> Self {
        match ftype {
            braid::FType::Continuous => Self::Continuous,
            braid::FType::Categorical => Self::Categorical,
            braid::FType::Count => Self::Count,
            _ => panic!("Labeler type nor supported"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Given {
    /// The conditions in `(column_id, value)` tuples. The tuple
    /// `(11, Datum::Continuous(2.3))` indicates that we wish to condition on
    /// the value of column 11 being 2.3.
    Conditions(Vec<(usize, Datum)>),
    /// The absence of conditioning observations
    Nothing,
}

impl Default for Given {
    fn default() -> Self {
        Self::Nothing
    }
}

impl From<Given> for braid::Given {
    fn from(given: Given) -> Self {
        match given {
            Given::Conditions(mut conditions) => braid::Given::Conditions(
                conditions.drain(..).map(|(ix, x)| (ix, x.into())).collect(),
            ),
            Given::Nothing => braid::Given::Nothing,
        }
    }
}

/// The type of uncertainty to use for `Oracle.impute`
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ImputeUncertaintyType {
    /// Given a set of distributions Θ = {Θ<sub>1</sub>, ..., Θ<sub>n</sub>},
    /// return the mean of KL(Θ<sub>i</sub> || Θ<sub>i</sub>)
    PairwiseKl,
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    JsDivergence,
}

impl From<ImputeUncertaintyType> for braid::ImputeUncertaintyType {
    fn from(unc_type: ImputeUncertaintyType) -> Self {
        match unc_type {
            ImputeUncertaintyType::PairwiseKl => Self::PairwiseKl,
            ImputeUncertaintyType::JsDivergence => Self::JsDivergence,
        }
    }
}

/// Mutual Information Type
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum MiType {
    /// The Standard, un-normalized variant
    UnNormed,
    /// Normalized by the max MI, which is `min(H(A), H(B))`
    Normed,
    /// Linfoot information Quantity. Derived by computing the mutual
    /// information between the two components of a bivariate Normal with
    /// covariance rho, and solving for rho.
    Linfoot,
    /// Variation of Information. A version of mutual information that
    /// satisfies the triangle inequality.
    Voi,
    /// Jaccard distance between X an Y. Jaccard(X, Y) is in [0, 1].
    Jaccard,
    /// Information Quality Ratio:  the amount of information of a variable
    /// based on another variable against total uncertainty.
    Iqr,
    /// Mutual Information normed the with square root of the product of the
    /// components entropies. Akin to the Pearson correlation coefficient.
    Pearson,
}

impl From<MiType> for braid::MiType {
    fn from(mi_type: MiType) -> Self {
        match mi_type {
            MiType::UnNormed => Self::UnNormed,
            MiType::Normed => Self::Normed,
            MiType::Linfoot => Self::Linfoot,
            MiType::Voi => Self::Voi,
            MiType::Jaccard => Self::Jaccard,
            MiType::Iqr => Self::Iqr,
            MiType::Pearson => Self::Pearson,
        }
    }
}

/// The type of uncertainty to use for `Oracle.predict`
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PredictUncertaintyType {
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    JsDivergence,
}

impl From<PredictUncertaintyType> for braid::PredictUncertaintyType {
    fn from(unc_type: PredictUncertaintyType) -> Self {
        match unc_type {
            PredictUncertaintyType::JsDivergence => Self::JsDivergence,
        }
    }
}

// /// Holds a `String` name or a `usize` index
// #[derive(Serialize, Deserialize, Component, Clone, Debug)]
// #[serde(untagged, rename_all = "snake_case")]
// pub enum NameOrIndex {
//     Name(String),
//     Index(usize),
// }

// #[derive(Serialize, Deserialize, Component, Clone, Debug)]
// #[serde(rename_all = "snake_case")]
// pub struct ColumnIndex(pub NameOrIndex);

// #[derive(Serialize, Deserialize, Component, Clone, Debug)]
// #[serde(rename_all = "snake_case")]
// pub struct RowIndex(pub NameOrIndex);

// #[derive(Serialize, Deserialize, Component, Clone, Debug)]
// pub struct Row {
//     /// The name of the row
//     pub row_ix: String,
//     /// The cells and values to fill in
//     pub values: Vec<Value>,
// }

// #[derive(Serialize, Deserialize, Component, Clone, Debug)]
// pub struct Value {
//     /// Name of the column
//     pub col_ix: String,
//     /// The value of the cell
//     pub value: Datum,
// }

// impl From<Value> for braid::Value<String> {
//     fn from(value: Value) -> Self {
//         Self {
//             col_ix: value.col_ix,
//             value: value.value,
//         }
//     }
// }

// impl From<Row> for braid::Row<String, String> {
//     fn from(mut row: Row) -> Self {
//         Self {
//             row_ix: row.row_ix,
//             values: r
//         }
//     }
// }

// impl Into<Row<String, String>

// impl From<NameOrIndex> for braid::NameOrIndex {
//     fn from(ix: NameOrIndex) -> Self {
//         match ix {
//             NameOrIndex::Name(name) => Self::Name(name),
//             NameOrIndex::Index(index) => Self::Index(index),
//         }
//     }
// }

// impl From<ColumnIndex> for braid::ColumnIndex {
//     fn from(ix: ColumnIndex) -> Self {
//         Self(ix.0.into())
//     }
// }

// impl From<RowIndex> for braid::RowIndex {
//     fn from(ix: RowIndex) -> Self {
//         Self(ix.0.into())
//     }
// }

// impl From<Value> for braid::Value {
//     fn from(value: Value) -> Self {
//         Self {
//             col_ix: value.col_ix.into(),
//             value: value.value.into(),
//         }
//     }
// }

// impl From<Row> for braid::Row {
//     fn from(mut row: Row) -> Self {
//         Self {
//             row_ix: row.row_ix.into(),
//             values: row.values.drain(..).map(|val| val.into()).collect(),
//         }
//     }
// }

/// Stores some diagnostic info in the `State` at every iteration
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
pub struct StateDiagnostics {
    /// Log likelihood
    #[serde(default)]
    pub loglike: Vec<f64>,
    /// Log prior likelihood
    #[serde(default)]
    pub log_prior: Vec<f64>,
    /// The number of views
    #[serde(default)]
    pub n_views: Vec<usize>,
    /// The state CRP alpha
    #[serde(default)]
    pub state_alpha: Vec<f64>,
    /// The number of categories in the views with the fewest categories
    #[serde(default)]
    pub n_cats_min: Vec<usize>,
    /// The number of categories in the views with the most categories
    #[serde(default)]
    pub n_cats_max: Vec<usize>,
    /// The median number of categories in a view
    #[serde(default)]
    pub n_cats_median: Vec<f64>,
}

impl From<braid::StateDiagnostics> for StateDiagnostics {
    fn from(diag: braid::StateDiagnostics) -> Self {
        Self {
            loglike: diag.loglike,
            /// Log prior likelihood
            log_prior: diag.logprior,
            /// The number of views
            n_views: diag.n_views,
            /// The state CRP alpha
            state_alpha: diag.state_alpha,
            /// The number of categories in the views with the fewest categories
            n_cats_min: diag.n_cats_min,
            /// The number of categories in the views with the most categories
            n_cats_max: diag.n_cats_max,
            /// The median number of categories in a view
            n_cats_median: diag.n_cats_median,
        }
    }
}

/// The MCMC algorithm to use for row reassignment
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    FiniteCpu,
    /// An Improved slice sampler based on stick breaking
    Slice,
    /// Sequential importance sampling split-merge
    Sams,
    /// Sequential, enumerative Gibbs
    Gibbs,
}

impl From<RowAssignAlg> for braid::cc::alg::RowAssignAlg {
    fn from(alg: RowAssignAlg) -> Self {
        match alg {
            RowAssignAlg::FiniteCpu => Self::FiniteCpu,
            RowAssignAlg::Gibbs => Self::Gibbs,
            RowAssignAlg::Slice => Self::Slice,
            RowAssignAlg::Sams => Self::Sams,
        }
    }
}

/// The MCMC algorithm to use for column reassignment
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ColAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    FiniteCpu,
    /// Sequential, enumerative Gibbs
    Gibbs,
    /// An Improved slice sampler based on stick breaking
    Slice,
}

impl From<ColAssignAlg> for braid::cc::alg::ColAssignAlg {
    fn from(alg: ColAssignAlg) -> Self {
        match alg {
            ColAssignAlg::FiniteCpu => Self::FiniteCpu,
            ColAssignAlg::Gibbs => Self::Gibbs,
            ColAssignAlg::Slice => Self::Slice,
        }
    }
}

/// MCMC transitions in the `State`
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum StateTransition {
    /// Reassign columns to views
    ColumnAssignment(ColAssignAlg),
    /// Reassign rows in views to categories
    RowAssignment(RowAssignAlg),
    /// Update the alpha (discount) parameter on the column-to-views CRP
    StateAlpha,
    /// Update the alpha (discount) parameters on the row-to-categories CRP
    ViewAlphas,
    /// Update the feature (column) prior parameters
    FeaturePriors,
    /// Update the parameters in the feature components. This is usually done
    /// automatically during the row assignment, but if the row assignment is
    /// not done (e.g. in the case of Geweke testing), then you can turn it on
    /// with this transition Note: this is not a default state transition.
    ComponentParams,
}

impl From<StateTransition> for braid::StateTransition {
    fn from(t: StateTransition) -> Self {
        match t {
            StateTransition::ColumnAssignment(alg) => {
                Self::ColumnAssignment(alg.into())
            }
            StateTransition::RowAssignment(alg) => {
                Self::RowAssignment(alg.into())
            }
            StateTransition::StateAlpha => Self::StateAlpha,
            StateTransition::ViewAlphas => Self::ViewAlphas,
            StateTransition::FeaturePriors => Self::FeaturePriors,
            StateTransition::ComponentParams => Self::ComponentParams,
        }
    }
}

#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum SummaryStatistics {
    Continuous {
        min: f64,
        max: f64,
        mean: f64,
        median: f64,
        variance: f64,
    },
    Categorical {
        min: u8,
        max: u8,
        mode: Vec<u8>,
    },
    Labeler {
        n: usize,
        n_true: usize,
        n_false: usize,
        n_labeled: usize,
        n_correct: usize,
    },
    Count {
        min: u32,
        max: u32,
        median: f64,
        mean: f64,
        mode: Vec<u32>,
    },
    None,
}

impl From<braid::SummaryStatistics> for SummaryStatistics {
    fn from(stats: braid::SummaryStatistics) -> Self {
        match stats {
            braid::SummaryStatistics::Continuous {
                min,
                max,
                mean,
                median,
                variance,
            } => Self::Continuous {
                min,
                max,
                mean,
                median,
                variance,
            },
            braid::SummaryStatistics::Categorical { min, max, mode } => {
                Self::Categorical { min, max, mode }
            }
            braid::SummaryStatistics::Labeler {
                n,
                n_true,
                n_false,
                n_labeled,
                n_correct,
            } => Self::Labeler {
                n,
                n_true,
                n_false,
                n_labeled,
                n_correct,
            },
            braid::SummaryStatistics::Count {
                min,
                max,
                median,
                mean,
                mode,
            } => Self::Count {
                min,
                max,
                median,
                mean,
                mode,
            },
            braid::SummaryStatistics::None => Self::None,
        }
    }
}

/// Defines insert data behavior -- where data may be inserted.
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum InsertMode {
    /// Can add new rows or column
    Unrestricted,
    /// Cannot add new rows, but can add new columns
    DenyNewRows,
    /// Cannot add new columns, but can add new rows
    DenyNewColumns,
    /// No adding new rows or columns
    DenyNewRowsAndColumns,
}

impl From<InsertMode> for braid::InsertMode {
    fn from(insert_mode: InsertMode) -> Self {
        match insert_mode {
            InsertMode::Unrestricted => Self::Unrestricted,
            InsertMode::DenyNewRows => Self::DenyNewRows,
            InsertMode::DenyNewColumns => Self::DenyNewColumns,
            InsertMode::DenyNewRowsAndColumns => Self::DenyNewRowsAndColumns,
        }
    }
}

/// Defines which data may be overwritten
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OverwriteMode {
    /// Overwrite anything
    Allow,
    /// Do not overwrite any existing cells. Only allow data in new rows or
    /// columns.
    Deny,
    /// Same as deny, but also allow existing cells that are empty to be
    /// overwritten.
    MissingOnly,
}

impl From<OverwriteMode> for braid::OverwriteMode {
    fn from(overwrite_mode: OverwriteMode) -> Self {
        match overwrite_mode {
            OverwriteMode::Allow => Self::Allow,
            OverwriteMode::Deny => Self::Deny,
            OverwriteMode::MissingOnly => Self::MissingOnly,
        }
    }
}

/// Defines the behavior of the data table when new rows are appended
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum AppendStrategy {
    /// New rows will be appended and the rest of the table will be unchanged
    None,
    /// If `n` rows are added, the top `n` rows will be removed
    Window,
    /// For each row added that exceeds `max_n_rows`, the row at `tench_ix` will
    /// be removed.
    Trench {
        /// The max number of rows allowed
        max_n_rows: usize,
        /// The index to remove data from
        trench_ix: usize,
    },
}

impl Default for AppendStrategy {
    fn default() -> Self {
        Self::None
    }
}

impl From<AppendStrategy> for braid::AppendStrategy {
    fn from(strat: AppendStrategy) -> Self {
        match strat {
            AppendStrategy::None => Self::None,
            AppendStrategy::Window => Self::Window,
            AppendStrategy::Trench {
                max_n_rows,
                trench_ix,
            } => Self::Trench {
                max_n_rows,
                trench_ix,
            },
        }
    }
}

/// Defines how/where data may be inserted, which day may and may not be
/// overwritten, and whether data may extend the domain
#[derive(Serialize, Deserialize, Component, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub struct WriteMode {
    /// Determines whether new rows or columns can be appended or if data may
    /// be entered into existing cells.
    pub insert: InsertMode,
    /// Determines if existing cells may or may not be overwritten or whether
    /// only missing cells may be overwritten.
    pub overwrite: OverwriteMode,
    /// If `true`, allow column support to be extended to accommodate new data
    /// that fall outside the range. For example, a binary column extends to
    /// ternary after the user inserts `Datum::Categorical(2)`.
    #[serde(default)]
    pub allow_extend_support: bool,
    /// The behavior of the table when new rows are appended
    #[serde(default)]
    pub append_strategy: AppendStrategy,
}

impl Default for WriteMode {
    fn default() -> Self {
        Self {
            insert: InsertMode::DenyNewRowsAndColumns,
            overwrite: OverwriteMode::MissingOnly,
            allow_extend_support: false,
            append_strategy: AppendStrategy::None,
        }
    }
}

impl From<WriteMode> for braid::WriteMode {
    fn from(write_mode: WriteMode) -> Self {
        Self {
            insert: write_mode.insert.into(),
            overwrite: write_mode.overwrite.into(),
            allow_extend_support: write_mode.allow_extend_support,
            append_strategy: write_mode.append_strategy.into(),
        }
    }
}

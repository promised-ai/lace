//! Stores mock versions of data objects in the main braid crate
use std::collections::BTreeMap;
use rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use braid_stats::prior::pg::PgHyper;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::labeler::{Labeler, LabelerPrior};
use braid_stats::prior::crp::CrpPrior;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Assignment {
    /// The `Crp` discount parameter
    pub alpha: f64,
    /// The assignment vector. `asgn[i]` is the partition index of the
    /// i<sup>th</sup> datum.
    pub asgn: Vec<usize>,
    /// Contains the number a data assigned to each partition
    pub counts: Vec<usize>,
    /// The number of partitions/categories
    pub ncats: usize,
    /// The prior on `alpha`
    pub prior: CrpPrior,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct StateDiagnostics {
    /// Log likelihood
    #[serde(default)]
    pub loglike: Vec<f64>,
    /// Log prior likelihood
    #[serde(default)]
    pub log_prior: Vec<f64>,
    /// The number of views
    #[serde(default)]
    pub nviews: Vec<usize>,
    /// The state CRP alpha
    #[serde(default)]
    pub state_alpha: Vec<f64>,
    /// The number of categories in the views with the fewest categories
    #[serde(default)]
    pub ncats_min: Vec<usize>,
    /// The number of categories in the views with the most categories
    #[serde(default)]
    pub ncats_max: Vec<usize>,
    /// The median number of categories in a view
    #[serde(default)]
    pub ncats_median: Vec<f64>,
}

impl Default for StateDiagnostics {
    fn default() -> Self {
        StateDiagnostics {
            loglike: vec![],
            log_prior: vec![],
            nviews: vec![],
            state_alpha: vec![],
            ncats_min: vec![],
            ncats_max: vec![],
            ncats_median: vec![],
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct DatalessState {
    views: Vec<DatalessView>,
    asgn: Assignment,
    weights: Vec<f64>,
    view_alpha_prior: CrpPrior,
    loglike: f64,
    #[serde(default)]
    log_prior: f64,
    #[serde(default)]
    log_view_alpha_prior: f64,
    #[serde(default)]
    log_state_alpha_prior: f64,
    diagnostics: StateDiagnostics,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct DatalessView {
    ftrs: BTreeMap<usize, DatalessColModel>,
    asgn: Assignment,
    weights: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
enum DatalessColModel {
    Continuous(DatalessColumn<Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(DatalessColumn<Categorical, SymmetricDirichlet, CsdHyper>),
    Labeler(DatalessColumn<Labeler, LabelerPrior, ()>),
    Count(DatalessColumn<Poisson, Gamma, PgHyper>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConjugateComponent<Fx, Stat> {
    pub fx: Fx,
    pub stat: Stat,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct DatalessColumn<Fx, Pr, H> {
    id: usize,
    components: Vec<ConjugateComponent<Fx, Pr>>,
    prior: Pr,
    hyper: H,
    #[serde(default)]
    ignore_hyper: bool,
}

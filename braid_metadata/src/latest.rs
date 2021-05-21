use crate::{impl_metadata_version, MetadataVersion};
use braid_codebook::Codebook;
use braid_data::SparseContainer;
use braid_stats::labeler::{Label, Labeler, LabelerPrior, LabelerSuffStat};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use rand_xoshiro::Xoshiro256Plus;
use rv::data::{CategoricalSuffStat, GaussianSuffStat, PoissonSuffStat};
use rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const METADATA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub states: Vec<DatalessState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_ids: Option<Vec<usize>>,
    pub codebook: Codebook,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataStore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng: Option<Xoshiro256Plus>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessState {
    pub views: Vec<DatalessView>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub view_alpha_prior: CrpPrior,
    pub loglike: f64,
    #[serde(default)]
    pub log_prior: f64,
    #[serde(default)]
    pub log_view_alpha_prior: f64,
    #[serde(default)]
    pub log_state_alpha_prior: f64,
    pub diagnostics: StateDiagnostics,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessView {
    pub ftrs: BTreeMap<usize, DatalessColModel>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub enum DatalessColModel {
    Continuous(
        DatalessColumn<
            Gaussian,
            NormalInvChiSquared,
            NixHyper,
            GaussianSuffStat,
        >,
    ),
    Categorical(
        DatalessColumn<
            Categorical,
            SymmetricDirichlet,
            CsdHyper,
            CategoricalSuffStat,
        >,
    ),
    Labeler(DatalessColumn<Labeler, LabelerPrior, (), LabelerSuffStat>),
    Count(DatalessColumn<Poisson, Gamma, PgHyper, PoissonSuffStat>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConjugateComponent<Fx, Stat> {
    pub fx: Fx,
    pub stat: Stat,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessColumn<Fx, Pr, H, Stat> {
    pub id: usize,
    pub components: Vec<ConjugateComponent<Fx, Stat>>,
    pub prior: Pr,
    pub hyper: H,
    #[serde(default)]
    pub ignore_hyper: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FeatureData {
    /// Univariate continuous data
    Continuous(SparseContainer<f64>),
    /// Categorical data
    Categorical(SparseContainer<u8>),
    /// Categorical data
    Labeler(SparseContainer<Label>),
    /// Count data
    Count(SparseContainer<u32>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DataStore(pub BTreeMap<usize, FeatureData>);

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

impl<Fx, Pr, H, Stat> MetadataVersion for DatalessColumn<Fx, Pr, H, Stat> {
    fn metadata_version() -> u32 {
        METADATA_VERSION
    }
}

impl<Fx, Stat> MetadataVersion for ConjugateComponent<Fx, Stat> {
    fn metadata_version() -> u32 {
        METADATA_VERSION
    }
}

impl_metadata_version!(DatalessColModel, METADATA_VERSION);
impl_metadata_version!(DatalessView, METADATA_VERSION);
impl_metadata_version!(DatalessState, METADATA_VERSION);
impl_metadata_version!(DataStore, METADATA_VERSION);
impl_metadata_version!(FeatureData, METADATA_VERSION);
impl_metadata_version!(Metadata, METADATA_VERSION);

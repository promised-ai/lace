//! Version 1 of the metadata
use std::collections::BTreeMap;

use lace_cc::assignment::Assignment;
use lace_cc::state::{State, StateDiagnostics};
use lace_cc::traits::{LaceDatum, LaceLikelihood, LaceStat};
use lace_data::FeatureData;
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};

use crate::{impl_metadata_version, MetadataVersion};

pub const METADATA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DataStore(BTreeMap<usize, FeatureData>);

impl From<lace_data::DataStore> for DataStore {
    fn from(data: lace_data::DataStore) -> Self {
        Self(data.0)
    }
}

impl From<DataStore> for lace_data::DataStore {
    fn from(data: DataStore) -> Self {
        Self(data.0)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PgHyper {
    pub pr_shape: Gamma,
    pub pr_rate: Gamma,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub enum ColType {
    Continuous {
        hyper: Option<NixHyper>,
        prior: Option<NormalInvChiSquared>,
    },
    Categorical {
        k: usize,
        hyper: Option<CsdHyper>,
        value_map: Option<BTreeMap<usize, String>>,
        prior: Option<SymmetricDirichlet>,
    },
    Count {
        hyper: Option<PgHyper>,
        prior: Option<Gamma>,
    },
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ColMetadata {
    pub name: String,
    pub coltype: ColType,
    pub notes: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Codebook {
    pub table_name: String,
    pub state_alpha_prior: Option<Gamma>,
    pub view_alpha_prior: Option<Gamma>,
    pub col_metadata: Vec<ColMetadata>,
    pub comments: Option<String>,
    pub row_names: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub states: Vec<DatalessState>,
    pub state_ids: Vec<usize>,
    pub codebook: Codebook,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataStore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng: Option<Xoshiro256Plus>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessState {
    pub views: Vec<DatalessView>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub view_alpha_prior: Gamma,
    pub loglike: f64,
    #[serde(default)]
    pub log_prior: f64,
    #[serde(default)]
    pub log_view_alpha_prior: f64,
    #[serde(default)]
    pub log_state_alpha_prior: f64,
    pub diagnostics: StateDiagnostics,
}

/// Marks a state as having no data in its columns
pub struct EmptyState(pub State);

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessView {
    pub ftrs: BTreeMap<usize, DatalessColModel>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub enum DatalessColModel {
    Continuous(DatalessColumn<f64, Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(DatalessColumn<u8, Categorical, SymmetricDirichlet, CsdHyper>),
    Count(DatalessColumn<u32, Poisson, Gamma, PgHyper>),
}

#[derive(Deserialize, Debug)]
pub struct ConjugateComponent<X, Fx>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
{
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub fx: Fx,
    #[serde(bound(deserialize = "Fx: serde::de::DeserializeOwned"))]
    pub stat: Fx::Stat,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DatalessColumn<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
{
    pub id: usize,
    #[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
    pub components: Vec<ConjugateComponent<X, Fx>>,
    #[serde(bound(deserialize = "Pr: serde::de::DeserializeOwned"))]
    pub prior: Pr,
    #[serde(bound(deserialize = "H: serde::de::DeserializeOwned"))]
    pub hyper: H,
    #[serde(default)]
    pub ignore_hyper: bool,
}

impl<X, Fx, Pr, H> MetadataVersion for DatalessColumn<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Fx::Stat: LaceStat,
{
    fn metadata_version() -> u32 {
        METADATA_VERSION
    }
}

impl_metadata_version!(PgHyper, METADATA_VERSION);
impl_metadata_version!(ColType, METADATA_VERSION);
impl_metadata_version!(ColMetadata, METADATA_VERSION);
impl_metadata_version!(Codebook, METADATA_VERSION);
impl_metadata_version!(DatalessColModel, METADATA_VERSION);
impl_metadata_version!(DatalessView, METADATA_VERSION);
impl_metadata_version!(DatalessState, METADATA_VERSION);
impl_metadata_version!(Metadata, METADATA_VERSION);

crate::loaders!(
    DatalessState,
    DataStore,
    Codebook,
    rand_xoshiro::Xoshiro256Plus
);

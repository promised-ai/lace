use braid::cc::config::EngineUpdateConfig;
use braid::cc::{ColAssignAlg, RowAssignAlg};
use braid::data::DataSource;
use braid::{Engine, EngineBuilder, Oracle};
use braid_codebook::codebook::Codebook;
use log::info;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum FeatureErrorDataset {
    #[serde(rename = "animals")]
    Animals { nstates: usize, n_iters: usize },
    #[serde(rename = "satellites")]
    Satellites { nstates: usize, n_iters: usize },
    #[serde(rename = "satellites-normed")]
    SatellitesNormed { nstates: usize, n_iters: usize },
}

impl FeatureErrorDataset {
    fn name(&self) -> &str {
        match self {
            FeatureErrorDataset::Animals { .. } => "animals",
            FeatureErrorDataset::Satellites { .. } => "satellites",
            FeatureErrorDataset::SatellitesNormed { .. } => "satellites-normed",
        }
    }

    fn nstates(&self) -> usize {
        match self {
            FeatureErrorDataset::Animals { nstates, .. } => *nstates,
            FeatureErrorDataset::Satellites { nstates, .. } => *nstates,
            FeatureErrorDataset::SatellitesNormed { nstates, .. } => *nstates,
        }
    }

    fn n_iters(&self) -> usize {
        match self {
            FeatureErrorDataset::Animals { n_iters, .. } => *n_iters,
            FeatureErrorDataset::Satellites { n_iters, .. } => *n_iters,
            FeatureErrorDataset::SatellitesNormed { n_iters, .. } => *n_iters,
        }
    }

    fn engine<R: Rng>(&self, nstates: usize, rng: &mut R) -> Engine {
        let mut dir = PathBuf::new();
        dir.push("resources");
        dir.push("datasets");
        dir.push(self.name());
        let dir = dir;

        let mut data_src = PathBuf::new();
        data_src.push(dir.clone());
        data_src.push(self.name());
        data_src.set_extension("csv");
        let data_src = data_src;

        let mut cb_src = dir.clone();
        cb_src.push(self.name());
        cb_src.set_extension("codebook.yaml");
        let cb_src = cb_src;

        let codebook = Codebook::from_yaml(&cb_src.as_path()).unwrap();
        EngineBuilder::new(DataSource::Csv(data_src))
            .with_nstates(nstates)
            .with_codebook(codebook)
            .with_seed(rng.next_u64())
            .build()
            .expect(format!("Couldn't build {} Engine", self.name()).as_str())
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct FeatureErrorResult {
    error: f64,
    centroid: f64,
}

impl FeatureErrorResult {
    fn new(pit_result: (f64, f64)) -> Self {
        FeatureErrorResult {
            error: pit_result.0,
            centroid: pit_result.1,
        }
    }
}

fn do_pit<R: Rng>(
    dataset: FeatureErrorDataset,
    mut rng: &mut R,
) -> Vec<FeatureErrorResult> {
    info!("Computing PITs for {} dataset", dataset.name());
    let mut engine = dataset.engine(dataset.nstates(), &mut rng);
    let config = EngineUpdateConfig::new()
        .with_iters(dataset.n_iters())
        .with_row_alg(RowAssignAlg::FiniteCpu)
        .with_col_alg(ColAssignAlg::Gibbs);

    engine.update(config);

    let oracle = Oracle::from_engine(engine);

    (0..oracle.ncols())
        .into_par_iter()
        .map(|col_ix| FeatureErrorResult::new(oracle.feature_error(col_ix)))
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PitRegressionConfig {
    pub datasets: Vec<FeatureErrorDataset>,
}

pub fn run_pit<R: Rng>(
    config: &PitRegressionConfig,
    mut rng: &mut R,
) -> BTreeMap<String, Vec<FeatureErrorResult>> {
    let mut results: BTreeMap<String, Vec<FeatureErrorResult>> =
        BTreeMap::new();
    config.datasets.iter().for_each(|&dataset| {
        let name = String::from(dataset.name());
        let res = do_pit(dataset, &mut rng);
        results.insert(name, res);
    });
    results
}

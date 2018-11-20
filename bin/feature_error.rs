extern crate braid;
extern crate rand;
extern crate serde_yaml;

use rayon::prelude::*;
use std::collections::BTreeMap;

use self::braid::cc::config::EngineUpdateConfig;
use self::braid::cc::{ColAssignAlg, RowAssignAlg};
use self::braid::data::DataSource;
use self::braid::{Codebook, Engine, EngineBuilder, Oracle};
use self::rand::{Rng, SeedableRng, XorShiftRng};

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

    fn engine<R: Rng>(&self, nstates: usize, mut rng: &mut R) -> Engine {
        let dir = format!("resources/datasets/{}", self.name());
        let data_src = format!("{}/{}.csv", dir, self.name());
        let cb_src = format!("{}/{}.codebook.yaml", dir, self.name());
        let codebook = Codebook::from_yaml(&cb_src);
        EngineBuilder::new(DataSource::Csv(data_src))
            .with_nstates(nstates)
            .with_codebook(codebook)
            .with_rng(XorShiftRng::from_rng(&mut rng).unwrap())
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

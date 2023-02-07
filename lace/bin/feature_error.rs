use lace::config::EngineUpdateConfig;
use lace::data::DataSource;
use lace::{Builder, Engine, Oracle, OracleT};

use lace_codebook::Codebook;
use log::info;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureErrorDataset {
    Animals { nstates: usize, n_iters: usize },
    Satellites { nstates: usize, n_iters: usize },
}

impl FeatureErrorDataset {
    fn name(&self) -> &str {
        match self {
            FeatureErrorDataset::Animals { .. } => "animals",
            FeatureErrorDataset::Satellites { .. } => "satellites",
        }
    }

    fn nstates(&self) -> usize {
        match self {
            FeatureErrorDataset::Animals { nstates, .. } => *nstates,
            FeatureErrorDataset::Satellites { nstates, .. } => *nstates,
        }
    }

    fn n_iters(&self) -> usize {
        match self {
            FeatureErrorDataset::Animals { n_iters, .. } => *n_iters,
            FeatureErrorDataset::Satellites { n_iters, .. } => *n_iters,
        }
    }

    fn engine<R: Rng>(&self, nstates: usize, rng: &mut R) -> Engine {
        let dir = {
            let mut dir = PathBuf::new();
            dir.push("resources");
            dir.push("datasets");
            dir.push(self.name());
            dir
        };

        // data source
        let data_src = {
            let mut data_src = PathBuf::new();
            data_src.push(dir.clone());
            data_src.push("data");
            data_src.set_extension("csv");
            data_src
        };

        // codebook source
        let cb_src = {
            let mut cb_src = dir;
            cb_src.push("codebook");
            cb_src.set_extension("yaml");
            cb_src
        };

        let codebook = Codebook::from_yaml(cb_src.as_path()).unwrap();
        Builder::new(DataSource::Csv(data_src))
            .with_nstates(nstates)
            .codebook(codebook)
            .seed_from_u64(rng.next_u64())
            .build()
            .unwrap_or_else(|_| panic!("Couldn't build {} Engine", self.name()))
    }
}

/// Contains the error and error centroid of the PIT
#[derive(Clone, Debug, Serialize)]
pub struct FeatureErrorResult {
    ///
    col_name: String,
    error: f64,
    centroid: f64,
}

fn do_pit<R: Rng>(
    dataset: &FeatureErrorDataset,
    mut rng: &mut R,
) -> Vec<FeatureErrorResult> {
    use lace::HasStates;

    info!("Computing PITs for {} dataset", dataset.name());
    let mut engine = dataset.engine(dataset.nstates(), &mut rng);
    let config = EngineUpdateConfig {
        n_iters: dataset.n_iters(),
        ..Default::default()
    };

    // shouldn't error because we're not doing any saving
    engine.update(config, None, None).unwrap();

    let oracle = Oracle::from_engine(engine);

    (0..oracle.n_cols())
        .into_par_iter()
        .map(|col_ix| {
            let col_name = oracle.codebook.col_metadata[col_ix].name.clone();
            let (error, centroid) = oracle.feature_error(col_ix).unwrap();
            FeatureErrorResult {
                col_name,
                error,
                centroid,
            }
        })
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PitRegressionConfig {
    pub datasets: Vec<FeatureErrorDataset>,
}

/// Run PIT/feature error. Computes the feature error using the probability
/// integral transform (PIT) for continuous data, and computes the error
/// between the true and empirical CDF for categorical.
pub fn run_pit<R: Rng>(
    config: &PitRegressionConfig,
    mut rng: &mut R,
) -> BTreeMap<String, Vec<FeatureErrorResult>> {
    let mut results: BTreeMap<String, Vec<FeatureErrorResult>> =
        BTreeMap::new();
    config.datasets.iter().for_each(|dataset| {
        let name = String::from(dataset.name());
        let res = do_pit(dataset, &mut rng);
        results.insert(name, res);
    });
    results
}

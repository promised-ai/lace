extern crate braid;
extern crate log;
extern crate rand;
extern crate serde;

use std::collections::BTreeMap;

use braid::cc::{state::StateGewekeSettings, State};
use braid::geweke::{GewekeResult, GewekeTester};
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct GewekeRegressionConfig {
    /// Config name and config
    pub settings: BTreeMap<String, StateGewekeSettings>,
    pub n_iters: usize,
    pub n_runs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag: Option<usize>,
}

#[derive(Serialize)]
pub struct GewekeRegressionResult {
    /// Replicate Geweke runs for various configurations.
    /// results["One"][1] is the 2nd Geweke run of the "One" config
    pub results: BTreeMap<String, Vec<GewekeResult>>,
    /// AUCs for geweke runs. Indexed the same as results
    pub aucs: BTreeMap<String, Vec<BTreeMap<String, f64>>>,
    // TODO: add timestamp
}

pub fn run_geweke<R: Rng>(
    config: &GewekeRegressionConfig,
    mut rng: &mut R,
) -> GewekeRegressionResult {
    let mut results = BTreeMap::new();
    let mut aucs = BTreeMap::new();
    for (name, cfg) in config.settings.iter() {
        info!("Running Geweke config '{}'", name);
        let cfg_res: Vec<GewekeResult> = (0..config.n_runs)
            .map(|i| {
                info!(
                    "Executing '{}' run {} of {}",
                    name,
                    i + 1,
                    config.n_runs
                );
                let mut gwk = GewekeTester::<State>::new(cfg.clone());
                gwk.verbose = true;
                gwk.run(config.n_iters, config.lag, &mut rng);
                gwk.result()
            })
            .collect();

        let cfg_aucs: Vec<BTreeMap<String, f64>> =
            cfg_res.iter().map(|r| r.aucs()).collect();

        results.insert(name.clone(), cfg_res);
        aucs.insert(name.clone(), cfg_aucs);
    }

    GewekeRegressionResult { results, aucs }
}

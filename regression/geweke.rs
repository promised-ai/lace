extern crate braid;
extern crate rand;

use self::braid::cc::{state::StateGewekeSettings, State};
use self::braid::geweke::{GewekeResult, GewekeTester};
use self::rand::Rng;
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct GewekeRegressionConfig {
    pub settings: StateGewekeSettings,
    pub n_iters: usize,
    pub n_runs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag: Option<usize>,
}

#[derive(Serialize)]
pub struct GewekeRegressionResult {
    pub results: Vec<GewekeResult>,
    pub aucs: Vec<BTreeMap<String, f64>>,
}

pub fn run_geweke<R: Rng>(
    config: &GewekeRegressionConfig,
    mut rng: &mut R,
) -> GewekeRegressionResult {
    let results: Vec<GewekeResult> = (0..config.n_runs)
        .map(|_| {
            let mut gwk = GewekeTester::<State>::new(config.settings.clone());
            gwk.verbose = true;
            gwk.run(config.n_iters, config.lag, &mut rng);
            gwk.result()
        })
        .collect();

    let aucs = results.iter().map(|r| r.aucs()).collect();

    GewekeRegressionResult {
        results: results,
        aucs: aucs,
    }
}

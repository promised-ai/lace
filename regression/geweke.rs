extern crate braid;
extern crate rand;

use self::braid::cc::{state::StateGewekeSettings, State};
use self::braid::geweke::{GewekeResult, GewekeTester};
use self::rand::Rng;

#[derive(Clone, Serialize, Deserialize)]
pub struct GewekeRegressionConfig {
    pub settings: Vec<StateGewekeSettings>,
    pub n_iters: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag: Option<usize>,
}

// TODO: revise geweke trait members to take Rng
pub fn run_geweke<R: Rng>(
    config: &GewekeRegressionConfig,
    mut rng: &mut R,
) -> Vec<GewekeResult> {
    config
        .settings
        .iter()
        .map(|s| {
            let mut geweke: GewekeTester<State> = GewekeTester::new(s.clone());
            geweke.verbose = true;
            geweke.run(config.n_iters, config.lag, &mut rng);
            geweke.result()
        })
        .collect()
}

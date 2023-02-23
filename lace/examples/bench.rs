use std::env;

use lace::bencher::Bencher;
use lace::prelude::*;
use lace_utils::{mean, std};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let nrows: usize = args[0].parse().unwrap_or(250);
    let ncols: usize = args[1].parse().unwrap_or(20);

    println!("Running benchmark on {} x {} states", nrows, ncols);

    let coltype = ColType::Categorical {
        k: 3,
        hyper: None,
        value_map: None,
        prior: None,
    };

    let state_buider = State::builder()
        .n_rows(nrows)
        .column_configs(ncols, coltype)
        .n_views(10)
        .n_cats(10);

    let config = StateUpdateConfig {
        transitions: vec![
            StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
            StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
            StateTransition::ComponentParams,
            StateTransition::FeaturePriors,
        ],
        ..Default::default()
    };

    let mut bencher = Bencher::from_builder(state_buider)
        .update_config(config)
        .n_iters(1)
        .n_runs(20);

    let mut rng = rand::thread_rng();

    let result = bencher.run(&mut rng);

    let times_sec: Vec<f64> =
        result.iter().map(|res| res.time_sec[0]).collect();

    let mean_time = mean(&times_sec);
    let std_time = std(&times_sec);

    println!("Time (M, S) = ({}s, {}s)", mean_time, std_time);
}

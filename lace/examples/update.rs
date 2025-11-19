//! Update and engine and show a progress bar
use std::time::Duration;

use lace::examples::Example;
use lace::update_handler::ProgressBar;
use lace::update_handler::Timeout;
use lace::EngineUpdateConfig;

fn main() {
    let mut engine = Example::Animals.engine().unwrap();

    let config = EngineUpdateConfig::new().default_transitions().n_iters(50);

    engine
        .update(
            config,
            (Timeout::new(Duration::from_secs(10)), ProgressBar::new()),
        )
        .unwrap();
}

//! Update and engine and show a progress bar
use lace::examples::Example;
use lace::misc::progress_bar;
use lace::update_handler::ProgressBar;
use lace::EngineUpdateConfig;

fn main() {
    let mut engine = Example::Animals.engine().unwrap();

    let config = EngineUpdateConfig::new()
        .default_transitions()
        .n_iters(50)
        .timeout(Some(10));

    engine.update(config, ProgressBar::new()).unwrap();
}

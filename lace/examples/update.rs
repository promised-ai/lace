//! Update and engine and show a progress bar
use lace::examples::Example;
use lace::misc::progress_bar;
use lace::EngineUpdateConfig;

fn main() {
    let mut engine = Example::Animals.engine().unwrap();

    let config = EngineUpdateConfig::new()
        .default_transitions()
        .n_iters(50)
        .timeout(Some(10));

    let (sndr, rcvr) = std::sync::mpsc::channel();

    let pbar_handle = progress_bar(engine.n_states() * config.n_iters, rcvr);

    engine.update(config, Some(sndr), None).unwrap();

    pbar_handle.join().expect("Failed to join progress bar");
}

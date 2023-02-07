//! Update and engine and show a progress bar
use lace::examples::Example;
use lace::misc::progress_bar;
use lace::{create_comms, EngineUpdateConfig};

#[tokio::main]
async fn main() {
    let mut engine = Example::Animals.engine().unwrap();

    let config = EngineUpdateConfig::new()
        .default_transitions()
        .n_iters(50)
        .timeout(Some(10));

    let (sndr, rcvr) = create_comms();

    let pbar_handle = progress_bar(engine.n_states() * config.n_iters, rcvr);

    let update_handle = tokio::spawn(async move {
        engine.update(config, Some(sndr), None).unwrap();
    });

    let (pbar_res, update_res) = tokio::join!(pbar_handle, update_handle);
    pbar_res.expect("Failed to join progress bar");
    update_res.expect("Failed to join update");
}

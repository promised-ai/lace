mod bench;
mod feature_error;
mod geweke;
mod opt;
mod regression;
mod routes;
mod shapes;
mod utils;

use clap::Parser;
use opt::Opt;

async fn route_cmd(opt: Opt) -> i32 {
    match opt {
        Opt::Codebook(cmd) => routes::codebook(cmd),
        Opt::Bench(cmd) => routes::bench(cmd),
        Opt::Regression(cmd) => regression::regression(cmd),
        Opt::Run(cmd) => routes::run(cmd).await,
        Opt::Summarize(cmd) => routes::summarize_engine(cmd),
        Opt::RegenExamples(cmd) => routes::regen_examples(cmd),
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let opt = Opt::parse();

    let exit_code = route_cmd(opt).await;

    std::process::exit(exit_code);
}

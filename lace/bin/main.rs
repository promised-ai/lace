#[cfg(feature = "dev")]
mod bench;
#[cfg(feature = "dev")]
mod feature_error;
#[cfg(feature = "dev")]
mod geweke;
mod opt;
#[cfg(feature = "dev")]
mod regression;
mod routes;
#[cfg(feature = "dev")]
mod shapes;
mod utils;

use clap::Parser;
use opt::Opt;

#[cfg(feature = "dev")]
async fn route_cmd(opt: Opt) -> i32 {
    match opt {
        Opt::Codebook(cmd) => routes::codebook(cmd),
        Opt::Bench(cmd) => routes::bench(cmd),
        Opt::Regression(cmd) => regression::regression(cmd),
        Opt::Run(cmd) => routes::run(cmd).await,
        Opt::Summarize(cmd) => routes::summarize_engine(cmd),
        Opt::RegenExamples(cmd) => routes::regen_examples(cmd),
        Opt::GenerateEncyrptionKey => routes::keygen(),
    }
}

#[cfg(not(feature = "dev"))]
async fn route_cmd(opt: Opt) -> i32 {
    match opt {
        Opt::Codebook(cmd) => routes::codebook(cmd),
        Opt::Run(cmd) => routes::run(cmd).await,
        Opt::Summarize(cmd) => routes::summarize_engine(cmd),
        Opt::GenerateEncyrptionKey => routes::keygen(),
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let opt = Opt::parse();

    let exit_code = route_cmd(opt).await;

    std::process::exit(exit_code);
}

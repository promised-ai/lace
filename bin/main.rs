mod bench;
mod feature_error;
mod geweke;
mod opt;
mod regression;
mod routes;
mod shapes;
mod utils;

use opt::Opt;
use structopt::StructOpt;

fn main() {
    env_logger::init();

    let opt = Opt::from_args();

    let exit_code: i32 = match opt {
        Opt::Codebook(cmd) => routes::codebook(cmd),
        Opt::Bench(cmd) => routes::bench(cmd),
        Opt::Regression(cmd) => regression::regression(cmd),
        Opt::Run(cmd) => routes::run(cmd),
        Opt::Summarize(cmd) => routes::summarize_engine(cmd),
        Opt::RegenExamples(cmd) => routes::regen_examples(cmd),
        Opt::GenerateEncyrptionKey => routes::keygen(),
    };

    std::process::exit(exit_code);
}

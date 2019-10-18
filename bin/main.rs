#![feature(rustc_private)]

mod bench;
mod braid_opt;
mod feature_error;
mod geweke;
mod regression;
mod routes;
mod shapes;

use crate::braid_opt::BraidOpt;
use structopt::StructOpt;

fn main() {
    env_logger::init();

    let opt = BraidOpt::from_args();

    let exit_code: i32 = match opt {
        BraidOpt::Append(cmd) => routes::append(cmd),
        BraidOpt::Codebook(cmd) => routes::codebook(cmd),
        BraidOpt::Bench(cmd) => routes::bench(cmd),
        BraidOpt::Regression(cmd) => regression::regression(cmd),
        BraidOpt::Run(cmd) => routes::run(cmd),
        BraidOpt::Summarize(cmd) => routes::summarize_engine(cmd),
        BraidOpt::RegenExamples => routes::regen_examples(),
    };

    std::process::exit(exit_code);
}

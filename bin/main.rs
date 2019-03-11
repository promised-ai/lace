#![feature(rustc_private)]

#[macro_use]
extern crate structopt;
#[macro_use]
extern crate serde;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate log;
#[macro_use]
extern crate maplit;

extern crate rayon;

mod bench;
mod braid_opt;
mod feature_error;
mod geweke;
mod regression;
mod routes;
mod shapes;

use braid_opt::BraidOpt;
use structopt::StructOpt;

fn main() {
    let opt = BraidOpt::from_args();

    let exit_code: i32 = match opt {
        BraidOpt::Append(cmd) => routes::append(cmd),
        BraidOpt::Codebook(cmd) => routes::codebook(cmd),
        BraidOpt::Bench(cmd) => routes::bench(cmd),
        BraidOpt::Regression(cmd) => regression::regression(cmd),
        BraidOpt::Run(cmd) => routes::run(cmd),
    };

    std::process::exit(exit_code);
}

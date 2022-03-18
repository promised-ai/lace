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

use opt::Opt;
use structopt::StructOpt;

#[cfg(feature = "idlock")]
const MACHINE_ID: &str = env!("BRAID_MACHINE_ID");

#[cfg(feature = "idlock")]
const LOCK_DATE: &str = env!("BRAID_LOCK_DATE", "");

#[cfg(feature = "idlock")]
fn validate_date() {
    use chrono::NaiveDate;

    if LOCK_DATE.is_empty() {
        return;
    }
    let lock_date = NaiveDate::parse_from_str(LOCK_DATE, "%Y-%m-%d").unwrap();
    let today = chrono::offset::Local::today().naive_local();

    if today > lock_date {
        panic!("License expired")
    }
}

#[cfg(not(feature = "idlock"))]
fn validate_machine_id() {}

#[cfg(feature = "idlock")]
fn validate_machine_id() {
    use rp_machine_id::{check_id, MachineIdVersion};
    check_id(MACHINE_ID, MachineIdVersion::V1).unwrap();
    validate_date()
}

#[cfg(feature = "dev")]
fn route_cmd(opt: Opt) -> i32 {
    match opt {
        Opt::Codebook(cmd) => routes::codebook(cmd),
        Opt::Bench(cmd) => routes::bench(cmd),
        Opt::Regression(cmd) => regression::regression(cmd),
        Opt::Run(cmd) => routes::run(cmd),
        Opt::Summarize(cmd) => routes::summarize_engine(cmd),
        Opt::RegenExamples(cmd) => routes::regen_examples(cmd),
        Opt::GenerateEncyrptionKey => routes::keygen(),
    }
}

#[cfg(not(feature = "dev"))]
fn route_cmd(opt: Opt) -> i32 {
    match opt {
        Opt::Codebook(cmd) => routes::codebook(cmd),
        Opt::Run(cmd) => routes::run(cmd),
        Opt::Summarize(cmd) => routes::summarize_engine(cmd),
        Opt::GenerateEncyrptionKey => routes::keygen(),
    }
}

fn main() {
    validate_machine_id();

    env_logger::init();

    let opt = Opt::from_args();

    let exit_code = route_cmd(opt);

    std::process::exit(exit_code);
}

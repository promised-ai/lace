use clap::Parser;
use rand_xoshiro::Xoshiro256Plus;

use lace::cc::alg::RowAssignAlg;
use lace::cc::feature::FType;
use lace::cc::transition::ViewTransition;
use lace::cc::view::{View, ViewGewekeSettings};
use lace::geweke::GewekeTester;
use rand::SeedableRng;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab")]
struct Opt {
    #[clap(long, default_value = "gibbs")]
    pub alg: RowAssignAlg,
    #[clap(short, long, default_value = "20")]
    pub nrows: usize,
    #[clap(long)]
    pub no_row_reassign: bool,
    #[clap(long)]
    pub no_view_alpha: bool,
    #[clap(long)]
    pub no_priors: bool,
    #[clap(long)]
    pub pitman_yor: bool,
    #[clap(long, short = 'i', default_value = "10000")]
    pub niters: usize,
}

fn main() {
    let opt = Opt::parse();

    println!("Running {} rows using {} algorithm", opt.nrows, opt.alg);

    let mut rng = Xoshiro256Plus::from_os_rng();
    let ftypes = vec![FType::Continuous, FType::Count, FType::Categorical];

    // The views's Geweke test settings require the number of rows in the
    // view (50), and the types of each column. Everything else is filled out
    // automatically.
    let settings = {
        let mut settings = ViewGewekeSettings::new(opt.nrows, ftypes);
        settings.transitions = Vec::new();
        if !opt.no_row_reassign {
            settings
                .transitions
                .push(ViewTransition::RowAssignment(opt.alg));
        }

        if !opt.no_view_alpha {
            settings
                .transitions
                .push(ViewTransition::PriorProcessParams);
        }

        if !opt.no_priors {
            settings.transitions.push(ViewTransition::FeaturePriors);
        }

        if opt.pitman_yor {
            settings = settings.with_pitman_yor_process();
        }

        settings.transitions.push(ViewTransition::ComponentParams);

        settings
    };

    // Initialize a tester given the settings and run.
    let mut geweke: GewekeTester<View> = GewekeTester::new(settings);
    geweke.run(opt.niters, Some(5), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    geweke.result().report();
}

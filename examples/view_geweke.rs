use braid::cc::transition::ViewTransition;
use braid::cc::view::ViewGewekeSettings;
use braid::cc::{FType, RowAssignAlg, View};
use braid_geweke::GewekeTester;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(
        long,
        default_value = "slice",
        possible_values = &["finite_cpu", "gibbs", "slice", "sams"],
    )]
    pub alg: RowAssignAlg,
    #[structopt(short, long, default_value = "20")]
    pub nrows: usize,
}

fn main() {
    let opt = Opt::from_args();

    println!("Running {} rows using {} algorithm", opt.nrows, opt.alg);

    let mut rng = Xoshiro256Plus::from_entropy();
    let ftypes = vec![FType::Continuous, FType::Count, FType::Categorical];

    // The views's Geweke test settings require the number of rows in the
    // view (50), and the types of each column. Everything else is filled out
    // automatically.
    let settings = {
        let mut settings = ViewGewekeSettings::new(opt.nrows, ftypes);
        settings.transitions = vec![
            ViewTransition::RowAssignment(opt.alg),
            ViewTransition::FeaturePriors,
            ViewTransition::ComponentParams,
            ViewTransition::Alpha,
        ];
        settings
    };

    // Initialize a tester given the settings and run.
    let mut geweke: GewekeTester<View> = GewekeTester::new(settings);
    geweke.run(10_000, Some(1), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    geweke.result().report();
}

use lace_cc::alg::RowAssignAlg;
use lace_cc::feature::FType;
use lace_cc::transition::ViewTransition;
use lace_cc::view::View;
use lace_cc::view::ViewGewekeSettings;
use lace_geweke::GewekeTester;
use clap::Parser;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab")]
struct Opt {
    #[clap(
        long,
        default_value = "gibbs",
        possible_values = &["finite_cpu", "gibbs", "slice", "sams"],
    )]
    pub alg: RowAssignAlg,
    #[clap(short, long, default_value = "20")]
    pub nrows: usize,
    #[clap(long)]
    pub no_row_reassign: bool,
    #[clap(long)]
    pub no_view_alpha: bool,
    #[clap(long)]
    pub no_priors: bool,
}

fn main() {
    let opt = Opt::parse();

    println!("Running {} rows using {} algorithm", opt.nrows, opt.alg);

    let mut rng = Xoshiro256Plus::from_entropy();
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
            settings.transitions.push(ViewTransition::Alpha);
        }
        if !opt.no_priors {
            settings.transitions.push(ViewTransition::FeaturePriors);
        }

        settings.transitions.push(ViewTransition::ComponentParams);

        settings
    };

    // Initialize a tester given the settings and run.
    let mut geweke: GewekeTester<View> = GewekeTester::new(settings);
    geweke.run(10_000, Some(5), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    geweke.result().report();
}

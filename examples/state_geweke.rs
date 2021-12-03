use braid_cc::alg::{ColAssignAlg, RowAssignAlg};
use braid_cc::feature::FType;
use braid_cc::state::State;
use braid_cc::state::StateGewekeSettings;
use braid_cc::transition::StateTransition;
use braid_geweke::GewekeTester;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(rename_all = "kebab")]
struct Opt {
    #[structopt(
        long,
        default_value = "gibbs",
        possible_values = &["finite_cpu", "gibbs", "slice", "sams"],
    )]
    pub row_alg: RowAssignAlg,
    #[structopt(
        long,
        default_value = "gibbs",
        possible_values = &["finite_cpu", "gibbs", "slice"],
    )]
    pub col_alg: ColAssignAlg,
    #[structopt(long, default_value = "50")]
    pub nrows: usize,
    #[structopt(long, default_value = "5")]
    pub ncols: usize,
    #[structopt(long)]
    pub no_row_reassign: bool,
    #[structopt(long)]
    pub no_col_reassign: bool,
    #[structopt(long)]
    pub no_view_alpha: bool,
    #[structopt(long)]
    pub no_state_alpha: bool,
    #[structopt(long)]
    pub no_priors: bool,
}

fn main() {
    let opt = Opt::from_args();
    let mut rng = Xoshiro256Plus::from_entropy();

    // Some of each column type that is supported by Geweke (Labeler cannot be
    // used in Geweke tests)
    let ftypes: Vec<FType> = (0..opt.ncols)
        .map(|i| match i % 3 {
            0 => FType::Continuous,
            1 => FType::Categorical,
            2 => FType::Count,
            _ => unreachable!(),
        })
        .collect();

    // The state's Geweke test settings require the number of rows in the
    // state (50), and the types of each column. Everything else is filled out
    // automatically.
    let mut settings = StateGewekeSettings::new(opt.nrows, ftypes);
    let mut transitions: Vec<StateTransition> = Vec::new();

    if !opt.no_col_reassign {
        transitions.push(StateTransition::ColumnAssignment(opt.col_alg));
    }

    if !opt.no_state_alpha {
        transitions.push(StateTransition::StateAlpha);
    }

    if !opt.no_row_reassign {
        transitions.push(StateTransition::RowAssignment(opt.row_alg));
    }

    if !opt.no_view_alpha {
        transitions.push(StateTransition::ViewAlphas);
    }

    if !opt.no_priors {
        transitions.push(StateTransition::FeaturePriors);
    }

    settings.transitions = transitions;

    // Initialize a tester given the settings and run.
    let mut geweke: GewekeTester<State> = GewekeTester::new(settings);
    geweke.run(10_000, Some(1), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    geweke.result().report();
}

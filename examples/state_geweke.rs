use braid::cc::state::StateGewekeSettings;
use braid::cc::transition::StateTransition;
use braid::cc::{ColAssignAlg, RowAssignAlg};
use braid::cc::{FType, State};
use braid_geweke::GewekeTester;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    let mut rng = Xoshiro256Plus::from_entropy();

    // Some of each column type that is supported by Geweke (Labeler cannot be
    // used in Geweke tests)
    let ftypes = vec![
        FType::Continuous,
        FType::Continuous,
        FType::Categorical,
        FType::Count,
        FType::Count,
    ];

    // The state's Geweke test settings require the number of rows in the
    // state (50), and the types of each column. Everything else is filled out
    // automatically.
    let mut settings = StateGewekeSettings::new(50, ftypes);
    settings.transitions = vec![
        StateTransition::ColumnAssignment(ColAssignAlg::Slice),
        StateTransition::StateAlpha,
        StateTransition::RowAssignment(RowAssignAlg::Slice),
        StateTransition::ViewAlphas,
        StateTransition::FeaturePriors,
    ];

    // Initialize a tester given the settings and run.
    let mut geweke: GewekeTester<State> = GewekeTester::new(settings);
    geweke.run(10_000, Some(1), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    geweke.result().report();
}

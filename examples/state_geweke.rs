extern crate braid;
extern crate rand;

use braid::cc::transition::StateTransition;
use braid::cc::FType;
use braid::cc::{state::StateGewekeSettings, State};
use braid::geweke::GewekeTester;
use rand::prng::XorShiftRng;
use rand::FromEntropy;

fn main() {
    let mut rng = XorShiftRng::from_entropy();
    let ftypes = vec![FType::Continuous; 5];

    // The state's Geweke test settings require the number of rows in the
    // state (50), and the types of each column. Everything else is filled out
    // automatically.
    let mut settings = StateGewekeSettings::new(50, ftypes);
    settings.transitions = vec![
        StateTransition::ColumnAssignment,
        StateTransition::StateAlpha,
        StateTransition::RowAssignment,
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

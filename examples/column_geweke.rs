extern crate braid;
extern crate rand;
extern crate serde_json;

use braid::cc::feature::ColumnGewekeSettings;
use braid::cc::transition::ViewTransition;
use braid::cc::{Assignment, AssignmentBuilder, Column};
use braid::dist::prior::{CatSymDirichlet, NormalInverseGamma};
use braid::dist::{Categorical, Gaussian};
use braid::geweke::*;

type ContinuousColumn = Column<f64, Gaussian, NormalInverseGamma>;
type CategoricalColumn = Column<u8, Categorical<u8>, CatSymDirichlet>;

fn main() {
    let mut rng = rand::thread_rng();

    // The column model uses an assignment as its setting. We'll draw a
    // 50-length assignment from the prior.
    let transitions = vec![
        ViewTransition::Alpha,
        // ViewTransition::FeaturePriors,
        ViewTransition::RowAssignment,
    ];
    let asgn = AssignmentBuilder::new(10).flat().build(&mut rng);

    let settings = ColumnGewekeSettings::new(asgn, transitions);

    // Initialize a tester for a continuous column model
    let mut geweke_cont: GewekeTester<ContinuousColumn> =
        GewekeTester::new(settings.clone());
    geweke_cont.run(10_000, Some(10), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    println!("Continuous");
    geweke_cont.result().report();
    // let result = geweke_cont.result();
    // let json = serde_json::to_string(&result).unwrap();
    // println!("{}", json)

    let mut geweke_cat: GewekeTester<CategoricalColumn> =
        GewekeTester::new(settings.clone());
    geweke_cat.run(1_000, Some(10), &mut rng);

    println!("\nCategorical");
    geweke_cat.result().report();
    // let result = geweke_cat.result();
    // let json = serde_json::to_string(&result).unwrap();
    // println!("{}", json)
}

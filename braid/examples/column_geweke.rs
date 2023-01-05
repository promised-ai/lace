use braid_geweke::*;
use braid_stats::prior::{csd::CsdHyper, nix::NixHyper};
use braid_stats::rv::dist::{
    Categorical, Gaussian, NormalInvChiSquared, SymmetricDirichlet,
};

use braid_cc::alg::RowAssignAlg;
use braid_cc::assignment::AssignmentBuilder;
use braid_cc::feature::geweke::ColumnGewekeSettings;
use braid_cc::feature::Column;
use braid_cc::transition::ViewTransition;

type ContinuousColumn = Column<f64, Gaussian, NormalInvChiSquared, NixHyper>;
type CategoricalColumn = Column<u8, Categorical, SymmetricDirichlet, CsdHyper>;

fn main() {
    let mut rng = rand::thread_rng();

    // The column model uses an assignment as its setting. We'll draw a
    // 50-length assignment from the prior.
    let transitions = vec![
        ViewTransition::Alpha,
        ViewTransition::RowAssignment(RowAssignAlg::Slice),
    ];
    let asgn = AssignmentBuilder::new(10).flat().build().unwrap();

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
        GewekeTester::new(settings);
    geweke_cat.run(1_000, Some(10), &mut rng);

    println!("\nCategorical");
    geweke_cat.result().report();
    // let result = geweke_cat.result();
    // let json = serde_json::to_string(&result).unwrap();
    // println!("{}", json)
}

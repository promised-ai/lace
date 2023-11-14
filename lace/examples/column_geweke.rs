use lace::prelude::*;
use lace_geweke::*;
use lace_stats::rv::dist::{
    Categorical, Gaussian, NormalInvChiSquared, SymmetricDirichlet,
};
use rand::Rng;

use lace_cc::feature::geweke::ColumnGewekeSettings;

type ContinuousColumn = Column<f64, Gaussian, NormalInvChiSquared, NixHyper>;
type CategoricalColumn = Column<u8, Categorical, SymmetricDirichlet, CsdHyper>;

#[cfg(not(feature = "experimental"))]
fn index_geweke(settings: ColumnGewekeSettings, mut rng: &mut impl Rng) {}

#[cfg(feature = "experimental")]
fn index_geweke(settings: ColumnGewekeSettings, mut rng: &mut impl Rng) {
    use lace_stats::experimental::sbd::SbdHyper;
    use lace_stats::rv::experimental::{Sb, Sbd};
    type IndexColumn = Column<usize, Sbd, Sb, SbdHyper>;

    let mut geweke_cat: GewekeTester<IndexColumn> = GewekeTester::new(settings);
    geweke_cat.run(1_000, Some(10), &mut rng);
    println!("\nIndex");
    geweke_cat.result().report();
}

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

    // // Initialize a tester for a continuous column model
    // let mut geweke_cont: GewekeTester<ContinuousColumn> =
    //     GewekeTester::new(settings.clone());
    // geweke_cont.run(10_000, Some(10), &mut rng);

    // // Reports the deviation from a perfect correspondence between the
    // // forward and posterior CDFs. The best score is zero, the worst possible
    // // score is 0.5.
    // println!("Continuous");
    // geweke_cont.result().report();
    // // let result = geweke_cont.result();
    // // let json = serde_json::to_string(&result).unwrap();
    // // println!("{}", json)

    let mut geweke_cat: GewekeTester<CategoricalColumn> =
        GewekeTester::new(settings.clone());
    geweke_cat.run(1_000, Some(10), &mut rng);

    println!("\nCategorical");
    geweke_cat.result().report();
    // let result = geweke_cat.result();
    // let json = serde_json::to_string(&result).unwrap();
    // println!("{}", json)

    index_geweke(settings, &mut rng);
}

#[macro_use]
extern crate approx;

use std::f64::consts::LN_2;

use braid_cc::assignment::{Assignment, AssignmentBuilder};
use braid_cc::component::ConjugateComponent;
use braid_cc::feature::{Column, Feature};
use braid_data::SparseContainer;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use once_cell::sync::OnceCell;
use rand::Rng;
use rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, SymmetricDirichlet,
};
use rv::traits::Rv;

type GaussCol = Column<f64, Gaussian, NormalInvChiSquared, NixHyper>;
type CatU8 = Column<u8, Categorical, SymmetricDirichlet, CsdHyper>;

fn gauss_fixture<R: Rng>(mut rng: &mut R, asgn: &Assignment) -> GaussCol {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let hyper = NixHyper::default();
    let data = SparseContainer::from(data_vec);
    let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);

    let mut col = Column::new(0, data, prior, hyper);
    col.reassign(&asgn, &mut rng);
    col
}

fn categorical_fixture_u8<R: Rng>(mut rng: &mut R, asgn: &Assignment) -> CatU8 {
    let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
    let data = SparseContainer::from(data_vec);
    let hyper = CsdHyper::vague(3);
    let prior = hyper.draw(3, &mut rng);

    let mut col = Column::new(0, data, prior, hyper);
    col.reassign(&asgn, &mut rng);
    col
}

fn three_component_column() -> GaussCol {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = SparseContainer::from(data_vec);
    let components = vec![
        ConjugateComponent::new(Gaussian::new(0.0, 1.0).unwrap()),
        ConjugateComponent::new(Gaussian::new(1.0, 1.0).unwrap()),
        ConjugateComponent::new(Gaussian::new(2.0, 1.0).unwrap()),
    ];

    let hyper = NixHyper::default();
    Column {
        id: 0,
        data,
        components,
        hyper,
        prior: NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0),
        ln_m_cache: OnceCell::new(),
        ignore_hyper: false,
    }
}

// Initialization
// ==============
#[test]
fn feature_with_flat_assign_should_have_one_component() {
    let mut rng = rand::thread_rng();
    let asgn = AssignmentBuilder::new(5).flat().build().unwrap();

    let col = gauss_fixture(&mut rng, &asgn);

    assert_eq!(col.components.len(), 1);
}

#[test]
fn feature_with_random_assign_should_have_k_component() {
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let asgn = AssignmentBuilder::new(5).build().unwrap();
        let col = gauss_fixture(&mut rng, &asgn);

        assert_eq!(col.components.len(), asgn.ncats);
    }
}

// Cleaning & book keeping
// =======================
#[test]
fn append_empty_component_appends_one() {
    let mut rng = rand::thread_rng();
    let asgn = AssignmentBuilder::new(5).flat().build().unwrap();
    let mut col = gauss_fixture(&mut rng, &asgn);

    assert_eq!(col.components.len(), 1);

    col.append_empty_component(&mut rng);

    assert_eq!(col.components.len(), 2);
}

#[test]
fn reassign_to_more_components() {
    let mut rng = rand::thread_rng();
    let asgn_a = AssignmentBuilder::new(5).flat().build().unwrap();
    let asgn_b = Assignment {
        alpha: 1.0,
        asgn: vec![0, 0, 0, 1, 1],
        counts: vec![3, 2],
        ncats: 2,
        prior: Gamma::new(1.0, 1.0).unwrap().into(),
    };

    let mut col = gauss_fixture(&mut rng, &asgn_a);

    assert_eq!(col.components.len(), 1);

    col.reassign(&asgn_b, &mut rng);

    assert_eq!(col.components.len(), 2);
}

#[test]
fn drop_middle_component() {
    let mut col = three_component_column();

    assert_eq!(col.components.len(), 3);
    assert_relative_eq!(col.components[0].fx.mu(), 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].fx.mu(), 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[2].fx.mu(), 2.0, epsilon = 10E-8);

    col.drop_component(1);

    assert_eq!(col.components.len(), 2);
    assert_relative_eq!(col.components[0].fx.mu(), 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].fx.mu(), 2.0, epsilon = 10E-8);
}

#[test]
fn drop_first_component() {
    let mut col = three_component_column();

    assert_eq!(col.components.len(), 3);
    assert_relative_eq!(col.components[0].fx.mu(), 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].fx.mu(), 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[2].fx.mu(), 2.0, epsilon = 10E-8);

    col.drop_component(0);

    assert_eq!(col.components.len(), 2);
    assert_relative_eq!(col.components[0].fx.mu(), 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].fx.mu(), 2.0, epsilon = 10E-8);
}

#[test]
fn drop_last_component() {
    let mut col = three_component_column();

    assert_eq!(col.components.len(), 3);
    assert_relative_eq!(col.components[0].fx.mu(), 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].fx.mu(), 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[2].fx.mu(), 2.0, epsilon = 10E-8);

    col.drop_component(2);

    assert_eq!(col.components.len(), 2);
    assert_relative_eq!(col.components[0].fx.mu(), 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].fx.mu(), 1.0, epsilon = 10E-8);
}

// Scores and accumulatiors
// ========================
//
// Gaussian
// --------
#[test]
fn gauss_accum_scores_1_cat_no_missing() {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = SparseContainer::from(data_vec);

    let hyper = NixHyper::default();
    let col = Column {
        id: 0,
        data,
        components: vec![ConjugateComponent::new(
            Gaussian::new(0.0, 1.0).unwrap(),
        )],
        hyper,
        prior: NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0),
        ln_m_cache: OnceCell::new(),
        ignore_hyper: false,
    };

    let mut scores: Vec<f64> = vec![0.0; 5];

    col.accum_score(&mut scores, 0);

    assert_relative_eq!(scores[0], -0.91893853320467267, epsilon = 10E-8);
    assert_relative_eq!(scores[1], -1.4189385332046727, epsilon = 10E-8);
    assert_relative_eq!(scores[2], -2.9189385332046727, epsilon = 10E-8);
    assert_relative_eq!(scores[3], -5.4189385332046722, epsilon = 10E-8);
    assert_relative_eq!(scores[4], -8.9189385332046722, epsilon = 10E-8);
}

#[test]
fn gauss_accum_scores_2_cats_no_missing() {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = SparseContainer::from(data_vec);
    let components = vec![
        ConjugateComponent::new(Gaussian::new(2.0, 1.0).unwrap()),
        ConjugateComponent::new(Gaussian::new(0.0, 1.0).unwrap()),
    ];

    let hyper = NixHyper::default();
    let col = Column {
        id: 0,
        data,
        components,
        hyper,
        prior: NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0),
        ln_m_cache: OnceCell::new(),
        ignore_hyper: false,
    };

    let mut scores: Vec<f64> = vec![0.0; 5];

    col.accum_score(&mut scores, 1);

    assert_relative_eq!(scores[0], -0.91893853320467267, epsilon = 10E-8);
    assert_relative_eq!(scores[1], -1.4189385332046727, epsilon = 10E-8);
    assert_relative_eq!(scores[2], -2.9189385332046727, epsilon = 10E-8);
    assert_relative_eq!(scores[3], -5.4189385332046722, epsilon = 10E-8);
    assert_relative_eq!(scores[4], -8.9189385332046722, epsilon = 10E-8);
}

#[test]
fn asgn_score_under_asgn_gaussian_magnitude() {
    let mut rng = rand::thread_rng();
    let asgn_a = AssignmentBuilder::new(5).flat().build().unwrap();
    let asgn_b = Assignment {
        alpha: 1.0,
        asgn: vec![0, 0, 0, 1, 1],
        counts: vec![3, 2],
        ncats: 2,
        prior: Gamma::new(1.0, 1.0).unwrap().into(),
    };

    let col = gauss_fixture(&mut rng, &asgn_a);

    let logp_a = col.asgn_score(&asgn_a);
    let logp_b = col.asgn_score(&asgn_b);

    // asgn_b should product a higher score because the data are increasing in
    // value. asgn_b encasultes the increasing data.
    assert!(logp_a < logp_b);
}

// Categorical
// -----------
#[test]
fn cat_u8_accum_scores_1_cat_no_missing() {
    let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
    let data = SparseContainer::from(data_vec);

    let log_weights = vec![
        -LN_2,               // log(0.5)
        -1.2039728043259361, // log(0.3)
        -1.6094379124341003,
    ]; // log(0.2)
    let col = Column {
        id: 0,
        data,
        components: vec![ConjugateComponent::new(
            Categorical::from_ln_weights(log_weights).unwrap(),
        )],
        prior: SymmetricDirichlet::new_unchecked(1.0, 3),
        hyper: CsdHyper::new(1.0, 1.0),
        ln_m_cache: OnceCell::new(),
        ignore_hyper: false,
    };

    let mut scores: Vec<f64> = vec![0.0; 5];

    col.accum_score(&mut scores, 0);

    assert_relative_eq!(scores[0], -0.6931471805599453, epsilon = 10E-8);
    assert_relative_eq!(scores[1], -1.2039728043259361, epsilon = 10E-8);
    assert_relative_eq!(scores[2], -1.6094379124341003, epsilon = 10E-8);
    assert_relative_eq!(scores[3], -0.6931471805599453, epsilon = 10E-8);
    assert_relative_eq!(scores[4], -1.2039728043259361, epsilon = 10E-8);
}

#[test]
fn cat_u8_accum_scores_2_cats_no_missing() {
    let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
    let data = SparseContainer::from(data_vec);

    let log_weights1 = vec![
        -LN_2,               // log(0.5)
        -1.2039728043259361, // log(0.3)
        -1.6094379124341003,
    ]; // log(0.2)
    let log_weights2 = vec![
        -1.2039728043259361, // log(0.3)
        -LN_2,               // log(0.5)
        -1.6094379124341003,
    ]; // log(0.2)

    let components = vec![
        ConjugateComponent::new(
            Categorical::from_ln_weights(log_weights1).unwrap(),
        ),
        ConjugateComponent::new(
            Categorical::from_ln_weights(log_weights2).unwrap(),
        ),
    ];
    let col = Column {
        id: 0,
        data,
        components,
        hyper: CsdHyper::new(1.0, 1.0),
        prior: SymmetricDirichlet::new_unchecked(1.0, 3),
        ln_m_cache: OnceCell::new(),
        ignore_hyper: false,
    };

    let mut scores: Vec<f64> = vec![0.0; 5];

    col.accum_score(&mut scores, 1);

    assert_relative_eq!(scores[0], -1.2039728043259361, epsilon = 10E-8);
    assert_relative_eq!(scores[1], -0.6931471805599453, epsilon = 10E-8);
    assert_relative_eq!(scores[2], -1.6094379124341003, epsilon = 10E-8);
    assert_relative_eq!(scores[3], -1.2039728043259361, epsilon = 10E-8);
    assert_relative_eq!(scores[4], -0.6931471805599453, epsilon = 10E-8);
}

#[test]
fn asgn_score_under_asgn_cat_u8_magnitude() {
    let mut rng = rand::thread_rng();
    let asgn_a = AssignmentBuilder::new(5).flat().build().unwrap();
    let asgn_b = Assignment {
        alpha: 1.0,
        asgn: vec![0, 1, 1, 0, 1],
        counts: vec![2, 3],
        ncats: 2,
        prior: Gamma::new(1.0, 1.0).unwrap().into(),
    };

    let col = categorical_fixture_u8(&mut rng, &asgn_a);

    let logp_a = col.asgn_score(&asgn_a);
    let logp_b = col.asgn_score(&asgn_b);

    // asgn_b should product a higher score because asgn_b groups partitions by
    // value
    assert!(logp_a < logp_b);
}

// Update component parameters
// ===========================
//
// Gaussian
// --------
#[test]
fn update_componet_params_should_draw_different_values_for_gaussian() {
    let mut rng = rand::thread_rng();
    let asgn = AssignmentBuilder::new(5).flat().build().unwrap();
    let mut col = gauss_fixture(&mut rng, &asgn);

    let cpnt_a = col.components[0].clone();
    col.update_components(&mut rng);
    let cpnt_b = col.components[0].clone();

    assert_relative_ne!(cpnt_a.fx.mu(), cpnt_b.fx.mu());
    assert_relative_ne!(cpnt_a.fx.sigma(), cpnt_b.fx.sigma());
}

#[test]
fn asgn_score_should_be_the_same_as_score_given_current_asgn() {
    let n = 100;
    let mut rng = rand::thread_rng();
    let g = Gaussian::standard();
    let hyper = NixHyper::default();
    for _ in 0..100 {
        let xs: Vec<f64> = g.sample(n, &mut rng);
        let data = SparseContainer::from(xs);
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);

        let mut col = Column::new(0, data, prior, hyper.clone());

        let asgn = AssignmentBuilder::new(n).flat().build().unwrap();
        let asgn_score = col.asgn_score(&asgn);
        col.reassign(&asgn, &mut rng);

        let score = col.score();

        assert_relative_eq!(score, asgn_score, epsilon = 1E-8);
    }
}

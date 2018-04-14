#[macro_use]
extern crate approx;

extern crate braid;
extern crate rand;
extern crate serde_test;

use self::rand::Rng;
use braid::cc::Assignment;
use braid::cc::Column;
use braid::cc::DataContainer;
use braid::cc::Feature;

use braid::dist::Gaussian;
use braid::dist::prior::csd::CsdHyper;
use braid::dist::prior::nig::NigHyper;
use braid::dist::prior::{CatSymDirichlet, NormalInverseGamma};

use braid::dist::Categorical;

type GaussCol = Column<f64, Gaussian, NormalInverseGamma>;
type CatU8 = Column<u8, Categorical<u8>, CatSymDirichlet>;

fn gauss_fixture(mut rng: &mut Rng, asgn: &Assignment) -> GaussCol {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let hyper = NigHyper::default();
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper);

    let mut col = Column::new(0, data, prior);
    col.reassign(&asgn, &mut rng);
    col
}

fn categorical_fixture_u8(mut rng: &mut Rng, asgn: &Assignment) -> CatU8 {
    let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
    let data = DataContainer::new(data_vec);
    let prior = CatSymDirichlet::vague(3, &mut rng);

    let mut col = Column::new(0, data, prior);
    col.reassign(&asgn, &mut rng);
    col
}

fn three_component_column() -> GaussCol {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);
    let components = vec![
        Gaussian::new(0.0, 1.0),
        Gaussian::new(1.0, 1.0),
        Gaussian::new(2.0, 1.0),
    ];

    let hyper = NigHyper::default();
    Column {
        id: 0,
        data: data,
        components: components,
        prior: NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper),
    }
}

// Initialization
// ==============
#[test]
fn feature_with_flat_assign_should_have_one_component() {
    let mut rng = rand::thread_rng();
    let asgn = Assignment::flat(5, 1.0);

    let col = gauss_fixture(&mut rng, &asgn);

    assert_eq!(col.components.len(), 1);
}

#[test]
fn feature_with_random_assign_should_have_k_component() {
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let asgn = Assignment::draw(5, 1.0, &mut rng);
        let col = gauss_fixture(&mut rng, &asgn);

        assert_eq!(col.components.len(), asgn.ncats);
    }
}

// Cleaning & book keeping
// =======================
#[test]
fn append_empty_component_appends_one() {
    let mut rng = rand::thread_rng();
    let asgn = Assignment::flat(5, 1.0);
    let mut col = gauss_fixture(&mut rng, &asgn);

    assert_eq!(col.components.len(), 1);

    col.append_empty_component(&mut rng);

    assert_eq!(col.components.len(), 2);
}

#[test]
fn reassign_to_more_components() {
    let mut rng = rand::thread_rng();
    let asgn_a = Assignment::flat(5, 1.0);
    let asgn_b = Assignment {
        alpha: 1.0,
        asgn: vec![0, 0, 0, 1, 1],
        counts: vec![3, 2],
        ncats: 2,
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
    assert_relative_eq!(col.components[0].mu, 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].mu, 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[2].mu, 2.0, epsilon = 10E-8);

    col.drop_component(1);

    assert_eq!(col.components.len(), 2);
    assert_relative_eq!(col.components[0].mu, 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].mu, 2.0, epsilon = 10E-8);
}

#[test]
fn drop_first_component() {
    let mut col = three_component_column();

    assert_eq!(col.components.len(), 3);
    assert_relative_eq!(col.components[0].mu, 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].mu, 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[2].mu, 2.0, epsilon = 10E-8);

    col.drop_component(0);

    assert_eq!(col.components.len(), 2);
    assert_relative_eq!(col.components[0].mu, 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].mu, 2.0, epsilon = 10E-8);
}

#[test]
fn drop_last_component() {
    let mut col = three_component_column();

    assert_eq!(col.components.len(), 3);
    assert_relative_eq!(col.components[0].mu, 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].mu, 1.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[2].mu, 2.0, epsilon = 10E-8);

    col.drop_component(2);

    assert_eq!(col.components.len(), 2);
    assert_relative_eq!(col.components[0].mu, 0.0, epsilon = 10E-8);
    assert_relative_eq!(col.components[1].mu, 1.0, epsilon = 10E-8);
}

// Scores and accumulatiors
// ========================
//
// Gaussian
// --------
#[test]
fn gauss_accum_scores_1_cat_no_missing() {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);

    let hyper = NigHyper::default();
    let col = Column {
        id: 0,
        data: data,
        components: vec![Gaussian::new(0.0, 1.0)],
        prior: NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper),
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
    let data = DataContainer::new(data_vec);
    let components = vec![Gaussian::new(2.0, 1.0), Gaussian::new(0.0, 1.0)];

    let hyper = NigHyper::default();
    let col = Column {
        id: 0,
        data: data,
        components: components,
        prior: NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper),
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
fn col_score_under_asgn_gaussian_magnitude() {
    let mut rng = rand::thread_rng();
    let asgn_a = Assignment::flat(5, 1.0);
    let asgn_b = Assignment {
        alpha: 1.0,
        asgn: vec![0, 0, 0, 1, 1],
        counts: vec![3, 2],
        ncats: 2,
    };

    let col = gauss_fixture(&mut rng, &asgn_a);

    let logp_a = col.col_score(&asgn_a);
    let logp_b = col.col_score(&asgn_b);

    // asgn_b should product a higher score because the data are increasing in
    // value. asgn_b encasultes the increasing data.
    assert!(logp_a < logp_b);
}

// Categorical
// -----------
#[test]
fn cat_u8_accum_scores_1_cat_no_missing() {
    let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
    let data = DataContainer::new(data_vec);

    let log_weights = vec![
        -0.6931471805599453, // log(0.5)
        -1.2039728043259361, // log(0.3)
        -1.6094379124341003,
    ]; // log(0.2)
    let col = Column {
        id: 0,
        data: data,
        components: vec![Categorical::new(log_weights)],
        prior: CatSymDirichlet::new(1.0, 3, CsdHyper::new(1.0, 1.0)),
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
    let data = DataContainer::new(data_vec);

    let log_weights1 = vec![
        -0.6931471805599453, // log(0.5)
        -1.2039728043259361, // log(0.3)
        -1.6094379124341003,
    ]; // log(0.2)
    let log_weights2 = vec![
        -1.2039728043259361, // log(0.3)
        -0.6931471805599453, // log(0.5)
        -1.6094379124341003,
    ]; // log(0.2)

    let components = vec![
        Categorical::new(log_weights1),
        Categorical::new(log_weights2),
    ];
    let col = Column {
        id: 0,
        data: data,
        components: components,
        prior: CatSymDirichlet::new(1.0, 3, CsdHyper::new(1.0, 1.0)),
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
fn col_score_under_asgn_cat_u8_magnitude() {
    let mut rng = rand::thread_rng();
    let asgn_a = Assignment::flat(5, 1.0);
    let asgn_b = Assignment {
        alpha: 1.0,
        asgn: vec![0, 1, 1, 0, 1],
        counts: vec![2, 3],
        ncats: 2,
    };

    let col = categorical_fixture_u8(&mut rng, &asgn_a);

    let logp_a = col.col_score(&asgn_a);
    let logp_b = col.col_score(&asgn_b);

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
    let asgn = Assignment::flat(5, 1.0);
    let mut col = gauss_fixture(&mut rng, &asgn);

    let cpnt_a = col.components[0].clone();
    col.update_components(&asgn, &mut rng);
    let cpnt_b = col.components[0].clone();

    assert_relative_ne!(cpnt_a.mu, cpnt_b.mu);
    assert_relative_ne!(cpnt_a.sigma, cpnt_b.sigma);
}

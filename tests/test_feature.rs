#[macro_use] extern crate approx;

extern crate rand;
extern crate braid;

use self::rand::Rng;
use braid::cc::Assignment;
use braid::cc::DataContainer;
use braid::cc::Feature;
use braid::cc::Column;
use braid::dist::Gaussian;
use braid::dist::prior::NormalInverseGamma;


type GaussCol<'a> = Column<'a, f64, Gaussian, NormalInverseGamma>;


fn fixture<'a>(mut rng: &mut Rng, asgn: &'a Assignment) -> GaussCol<'a> {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);

    Column::new(data, &asgn, prior, &mut rng)
}


#[test]
fn gauss_feature_with_flat_assign_should_have_one_component() {
    let mut rng = rand::thread_rng();
    let asgn = Assignment::flat(5, 1.0);

    let col = fixture(&mut rng, &asgn);

    assert_eq!(col.components.len(), 1);
}


#[test]
fn gauss_feature_with_random_assign_should_have_k_component() {
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let asgn = Assignment::draw(5, 1.0, &mut rng);
        let col = fixture(&mut rng, &asgn);

        assert_eq!(col.components.len(), asgn.ncats);
    }
}


#[test]
fn append_empty_component_appends_one() {
    let mut rng = rand::thread_rng();
    let asgn = Assignment::flat(5, 1.0);
    let mut col = fixture(&mut rng, &asgn);
    
    assert_eq!(col.components.len(), 1);

    col.append_empty_component(&mut rng);

    assert_eq!(col.components.len(), 2);
}

#[test]
fn accum_scores_1_cat_no_missing() {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);
    let asgn = Assignment::flat(5, 1.0);

    let col = Column{data: data,
                      asgn: &asgn,
                      components: vec![Gaussian::new(0.0, 1.0)],
                      prior: NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0)};

    let mut scores: Vec<f64> = vec![0.0; 5];

    col.accum_score(&mut scores, 0);

    relative_eq!(scores[0], -0.91893853320467267, epsilon = 10E-8);
    relative_eq!(scores[1], -1.4189385332046727, epsilon = 10E-8);
    relative_eq!(scores[2], -2.9189385332046727, epsilon = 10E-8);
    relative_eq!(scores[3], -5.4189385332046722, epsilon = 10E-8);
    relative_eq!(scores[4], -8.9189385332046722, epsilon = 10E-8);
}


#[test]
fn accum_scores_2_cats_no_missing() {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);
    let asgn = Assignment::flat(5, 1.0);
    let components = vec![Gaussian::new(2.0, 1.0), Gaussian::new(0.0, 1.0)];

    let col = Column{data: data,
                     asgn: &asgn,
                     components: components,
                     prior: NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0)};

    let mut scores: Vec<f64> = vec![0.0; 5];

    col.accum_score(&mut scores, 1);

    relative_eq!(scores[0], -0.91893853320467267, epsilon = 10E-8);
    relative_eq!(scores[1], -1.4189385332046727, epsilon = 10E-8);
    relative_eq!(scores[2], -2.9189385332046727, epsilon = 10E-8);
    relative_eq!(scores[3], -5.4189385332046722, epsilon = 10E-8);
    relative_eq!(scores[4], -8.9189385332046722, epsilon = 10E-8);
}


#[test]
fn update_componet_params_should_draw_different_values_for_gaussian() {
    let mut rng = rand::thread_rng();
    let asgn = Assignment::flat(5, 1.0);
    let mut col = fixture(&mut rng, &asgn);

    let cpnt_a = col.components[0].clone();
    col.update_component_params(&mut rng);
    let cpnt_b = col.components[0].clone();

    relative_ne!(cpnt_a.mu, cpnt_b.mu);
    relative_ne!(cpnt_a.sigma, cpnt_b.sigma);
}


#[test]
fn reassign_to_more_components() {
    let mut rng = rand::thread_rng();
    let asgn_a = Assignment::flat(5, 1.0);
    let asgn_b = Assignment{alpha: 1.0,
                            asgn: vec![0, 0, 0, 1, 1],
                            counts: vec![3, 2],
                            ncats: 2};

    let mut col = fixture(&mut rng, &asgn_a);

    assert_eq!(col.components.len(), 1);

    col.reassign(&asgn_b, &mut rng); 

    assert_eq!(col.components.len(), 2);
}


#[test]
fn reassign_to_fewer_components() {
    let mut rng = rand::thread_rng();
    let asgn_a = Assignment{alpha: 1.0,
                            asgn: vec![0, 0, 0, 1, 1],
                            counts: vec![3, 2],
                            ncats: 2};
    let asgn_b = Assignment::flat(5, 1.0);

    let mut col = fixture(&mut rng, &asgn_a);

    assert_eq!(col.components.len(), 2);

    col.reassign(&asgn_b, &mut rng); 

    assert_eq!(col.components.len(), 1);
}


#[test]
fn col_score_under_asgn_gaussian_magnitude() {
    let mut rng = rand::thread_rng();
    let asgn_a = Assignment::flat(5, 1.0);
    let asgn_b = Assignment{alpha: 1.0,
                            asgn: vec![0, 0, 0, 1, 1],
                            counts: vec![3, 2],
                            ncats: 2};

    let mut col = fixture(&mut rng, &asgn_a);

    let logp_a = col.col_score();
    let logp_b = col.col_score_under_asgn(&asgn_b);

    // asgn_b should product a higher score because the data are increasing in
    // value. asgn_b encasultes the increasing data.
    assert!(logp_a < logp_b);
}

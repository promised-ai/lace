extern crate rand;
extern crate braid;

use braid::cc::Assignment;
use braid::cc::DataContainer;
use braid::cc::Feature;
use braid::dist::prior::{NormalInverseGamma, NigHyper};


#[test]
fn gauss_feature_with_flat_assign_should_have_one_component() {
    let mut rng = rand::thread_rng();
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);
    let asgn = Assignment::flat(5, 1.0);

    let ftr = Feature::new(data, &asgn, prior, &mut rng);

    assert_eq!(ftr.components.len(), 1);
}


#[test]
fn gauss_feature_with_random_assign_should_have_k_component() {
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let data = DataContainer::new(data_vec);
        let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);
        let asgn = Assignment::draw(5, 1.0, &mut rng);
        let ftr = Feature::new(data, &asgn, prior, &mut rng);

        assert_eq!(ftr.components.len(), asgn.ncats);
    }
}

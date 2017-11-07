extern crate rand;
extern crate braid;

use braid::cc::Assignment;
use braid::cc::DataContainer;
use braid::cc::Feature;
use braid::dist::prior::NormalInverseGamma;

#[test]
fn create_gaussian_feature() {
    let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let data = DataContainer::new(data_vec);
    let asgn = Assignment::flat(5, 1.0);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);

    let mut rng = rand::thread_rng();

    let ftr = Feature::new(data, &asgn, prior, &mut rng); 
}

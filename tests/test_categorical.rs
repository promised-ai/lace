#[macro_use] extern crate assert_approx_eq;

extern crate braid;

use braid::dist::Categorical;

const TOL: f64 = 1E-8; 


#[test]
fn categorical_new() {
    let ctgrl: Categorical<u8> = Categorical::new(vec![0.0, 0.1, 0.2]);

    assert_approx_eq!(ctgrl.log_weights[0], -1.20194285, TOL);
    assert_approx_eq!(ctgrl.log_weights[1], -1.10194285, TOL);
    assert_approx_eq!(ctgrl.log_weights[2], -1.00194285, TOL);
}

#[test]
fn categorical_flat() {
    let ctgrl: Categorical<u8> = Categorical::flat(3);

    assert_eq!(ctgrl.log_weights.len(), 3);

    assert_approx_eq!(ctgrl.log_weights[0], ctgrl.log_weights[1], TOL);
    assert_approx_eq!(ctgrl.log_weights[1], ctgrl.log_weights[2], TOL);
}

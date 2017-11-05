#[macro_use] extern crate assert_approx_eq;

extern crate braid;

use braid::special::gamma;
use braid::special::gammaln;


// gamma function
// --------------
#[test]
fn gamma_1_should_be_1() {
    assert_approx_eq!(gamma(1.0), 1.0, 1E-10);
}

#[test]
fn gamma_2_should_be_1() {
    assert_approx_eq!(gamma(2.0), 1.0, 1E-10);
}

#[test]
fn gamma_small_value_test_1() {
    assert_approx_eq!(gamma(0.00099), 1009.52477271, 1E-3);
}

#[test]
fn gamma_small_value_test_2() {
    assert_approx_eq!(gamma(0.00100), 999.423772485, 1E-4);
}

#[test]
fn gamma_small_value_test_3() {
    assert_approx_eq!(gamma(0.00101), 989.522792258, 1E-4);
}

#[test]
fn gamma_medium_value_test_1() {
    assert_approx_eq!(gamma(6.1), 142.451944066, 1E-6);
}

#[test]
fn gamma_medium_value_test_2() {
    assert_approx_eq!(gamma(11.999), 39819417.4793, 1E-3);
}

#[test]
fn gamma_large_value_test_1() {
    assert_approx_eq!(gamma(12.0), 39916800.0, 1E-3);
}

#[test]
fn gamma_large_value_test_2() {
    assert_approx_eq!(gamma(12.001), 40014424.1571, 1E-3);
}

#[test]
fn gamma_large_value_test_3() {
    assert_approx_eq!(gamma(15.2), 149037380723.0, 1.0);
}


// gammaln (log gamma)
// -------------------
#[test]
fn gammaln_small_value_test_1() {
    assert_approx_eq!(gammaln(0.9999), 5.77297915613e-05, 1E-5);
}

#[test]
fn gammaln_small_value_test_2() {
    assert_approx_eq!(gammaln(1.0001), -5.77133422205e-05, 1E-5);
}

#[test]
fn gammaln_medium_value_test_1() {
    assert_approx_eq!(gammaln(3.1), 0.787375083274, 1E-5);
}

#[test]
fn gammaln_medium_value_test_2() {
    assert_approx_eq!(gammaln(6.3), 5.30734288962, 1E-5);
}

#[test]
fn gammaln_medium_value_test_3() {
    assert_approx_eq!(gammaln(11.9999), 17.5020635801, 1E-5);
}

#[test]
fn gammaln_large_value_test_1() {
    assert_approx_eq!(gammaln(12.0), 17.5023078459, 1E-5);
}

#[test]
fn gammaln_large_value_test_2() {
    assert_approx_eq!(gammaln(12.0001), 17.5025521125, 1E-5);
}

#[test]
fn gammaln_large_value_test_3() {
    assert_approx_eq!(gammaln(27.4), 62.5755868211, 1E-5);
}

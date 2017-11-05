#[macro_use] extern crate assert_approx_eq;

extern crate braid;

use braid::special::erf;
use braid::special::erfinv;


// erf (error function)
// --------------------
#[test]
fn erf_of_very_large_value_should_be_1() {
    assert_approx_eq!(erf(10.0), 1.0, 1E-7);
}

#[test]
fn erf_of_very_small_negative_value_should_be_negative_1() {
    assert_approx_eq!(erf(-10.0), -1.0, 1E-7);
}

#[test]
fn erf_of_zero_should_be_zero() {
    assert_approx_eq!(erf(0.0), 0.0, 1E-7);
}

#[test]
fn erf_negative_value_test_1() {
    assert_approx_eq!(erf(-0.25), -0.2763263901682369, 1E-6);
}

#[test]
fn erf_negative_value_test_2() {
    assert_approx_eq!(erf(-1.0), -0.84270079294971478, 1E-6);
}

#[test]
fn erf_positive_value_test_1() {
    assert_approx_eq!(erf(0.446), 0.47178896844758522, 1E-6);
}

#[test]
fn erf_positive_value_test_2() {
    assert_approx_eq!(erf(0.001), 0.0011283787909692363, 1E-6);
}


// inverf (inverse error function)
// -------------------------------
#[test]
fn erfinv_of_zero_should_be_zero() {
    assert_approx_eq!(erfinv(0.0), 0.0, 1E-7);
}

#[test]
fn erfinv_positive_value_test_1() {
    assert_approx_eq!(erfinv(0.5), 0.47693627620446982, 1E-6);
}

#[test]
fn erfinv_positive_value_test_2() {
    assert_approx_eq!(erfinv(0.121), 0.10764782605515244, 1E-6);
}

#[test]
fn erfinv_negative_value_test_1() {
    assert_approx_eq!(erfinv(-0.99), -1.8213863677184492, 1E-5);
}

#[test]
fn erfinv_negative_value_test_2() {
    assert_approx_eq!(erfinv(-0.999), -2.3267537655135242, 1E-4);
}

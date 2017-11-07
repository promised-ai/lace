#[macro_use] extern crate assert_approx_eq;

extern crate rand;
extern crate braid;

use braid::dist::Bernoulli;
use braid::dist::traits::Distribution;
use braid::dist::traits::Moments;


#[test]
fn mean() {
    let bern = Bernoulli::new(0.6);
    assert_approx_eq!(bern.mean(), 0.6, 10E-8);
}


#[test]
fn variance() {
    let bern = Bernoulli::new(0.3);
    assert_approx_eq!(bern.var(), 0.3 * 0.7, 10E-8);
}


#[test]
fn like_true_should_be_log_of_p() {
    let bern1 = Bernoulli::new(0.5);
    assert_approx_eq!(bern1.like(&true), 0.5, 10E-8);

    let bern2 = Bernoulli::new(0.95);
    assert_approx_eq!(bern2.like(&true), 0.95, 10E-8);
}


#[test]
fn like_false_should_be_log_of_q() {
    let bern1 = Bernoulli::new(0.5);
    assert_approx_eq!(bern1.like(&false), 0.5, 10E-8);

    let bern2 = Bernoulli::new(0.95);
    assert_approx_eq!(bern2.like(&false), 0.05, 10E-8);
}


#[test]
fn loglike_true_should_be_log_of_p() {
    let bern1 = Bernoulli::new(0.5);
    assert_approx_eq!(bern1.loglike(&true), (0.5 as f64).ln(), 10E-8);

    let bern2 = Bernoulli::new(0.95);
    assert_approx_eq!(bern2.loglike(&true), (0.95 as f64).ln(), 10E-8);
}


#[test]
fn loglike_false_should_be_log_of_q() {
    let bern1 = Bernoulli::new(0.5);
    assert_approx_eq!(bern1.loglike(&false), (0.5 as f64).ln(), 10E-8);

    let bern2 = Bernoulli::new(0.95);
    assert_approx_eq!(bern2.loglike(&false), (0.05 as f64).ln(), 10E-8);
}

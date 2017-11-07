#[macro_use] extern crate assert_approx_eq;

extern crate rand;
extern crate braid;

use braid::dist::Gaussian;
use braid::dist::traits::Distribution;
use braid::dist::traits::RandomVariate;
use braid::dist::traits::Moments;


const TOL: f64 = 1E-8; 


#[test]
fn gaussian_new() {
    let gauss = Gaussian::new(1.2, 3.0);

    assert_approx_eq!(gauss.mu, 1.2, TOL);
    assert_approx_eq!(gauss.sigma, 3.0, TOL);
}


#[test]
fn gaussian_standard() {
    let gauss = Gaussian::standard();

    assert_approx_eq!(gauss.mu, 0.0, TOL);
    assert_approx_eq!(gauss.sigma, 1.0, TOL);
}


#[test]
fn gaussian_moments() {
    let gauss1 = Gaussian::standard();

    assert_approx_eq!(gauss1.mean(), 0.0, TOL);
    assert_approx_eq!(gauss1.var(), 1.0, TOL);

    let gauss2 = Gaussian::new(3.4, 0.5);

    assert_approx_eq!(gauss2.mean(), 3.4, TOL);
    assert_approx_eq!(gauss2.var(), 0.25, TOL);
}


#[test]
fn gaussian_sample_length() {
    let mut rng = rand::thread_rng();
    let gauss = Gaussian::standard();
    let xs: Vec<f64> = gauss.sample(10, &mut rng);
    assert_eq!(xs.len(), 10);
}


#[test]
fn gaussian_standard_loglike() {
    let gauss = Gaussian::standard();
    assert_approx_eq!(gauss.loglike(&0.0), -0.91893853320467267, TOL);
    assert_approx_eq!(gauss.loglike(&2.1), -3.1239385332046727, TOL);
}


#[test]
fn gaussian_nonstandard_loglike() {
    let gauss = Gaussian::new(-1.2, 0.33);

    assert_approx_eq!(gauss.loglike(&-1.2), 0.18972409131693846, TOL);
    assert_approx_eq!(gauss.loglike(&0.0), -6.4218461566169447, TOL);
}

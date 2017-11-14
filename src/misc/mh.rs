extern crate rand;

use std::f64;
use self::rand::Rng;
use self::rand::distributions::{Range, Normal, IndependentSample};


fn mh_prior<T, F, D>(loglike: F, prior_draw: D, n_iter: usize,
                     mut rng: &mut Rng) -> T
    where F: Fn(&T) -> f64,
          D: Fn(&mut Rng) -> T,
{
    let u = Range::new(0.0, 1.0);
    let mut x: T = prior_draw(&mut rng);
    let mut score: f64 = loglike(&x);
    for _ in 0..n_iter {
        let x_r = prior_draw(&mut rng);
        let score_r = loglike(&x_r);
        let r: f64 = u.ind_sample(&mut rng);
        if r.ln() < score_r - score {
            score = score_r;
            x = x_r;
        }
    }
    x
}

// TODO: Random Walk

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mh_prior_uniform() {
        let range = Range::new(0.0, 1.0);

        let loglike = |x: &f64| { 0.0 };
        let prior_draw = |mut r: &mut Rng| { range.ind_sample(&mut r) };

        let mut rng = rand::thread_rng();
        let x = mh_prior(loglike, prior_draw, 25, &mut rng);

        assert!(x <= 1.0);
        assert!(x >= 0.0);
    }


    #[test]
    fn test_mh_prior_normal() {
        let norm = Normal::new(0.0, 1.0);

        let loglike = |x: &f64| { 0.0 };
        let prior_draw = |mut r: &mut Rng| { norm.ind_sample(&mut r) };

        let mut rng = rand::thread_rng();
        let x = mh_prior(loglike, prior_draw, 25, &mut rng);
    }
}

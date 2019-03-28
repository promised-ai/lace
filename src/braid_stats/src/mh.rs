extern crate rand;

use std::f64;

use rand::distributions::Uniform;
use rand::Rng;

/// Draw posterior samples from f(x|y)π(x) by taking proposals from the prior
///
/// # Arguments
/// - x_start: the starting value
/// - loglike: the liklihood function, f(y|x)
/// - prior_draw: the draw function of the prior on `x`
/// - n_iters: the number of MH steps
/// - rng: The random number generator
pub fn mh_prior<T, F, D, R: Rng>(
    x_start: T,
    loglike: F,
    prior_draw: D,
    n_iters: usize,
    mut rng: &mut R,
) -> T
where
    F: Fn(&T) -> f64,
    D: Fn(&mut R) -> T,
{
    let u = Uniform::new(0.0, 1.0);
    let x = x_start;
    let fx = loglike(&x);
    (0..n_iters)
        .fold((x, fx), |(x, fx), _| {
            let y = prior_draw(&mut rng);
            let fy = loglike(&y);
            let r: f64 = rng.sample(u);
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .0
}

// TODO: Random Walk

#[cfg(test)]
mod tests {
    use self::rand::distributions::Normal;
    use super::*;

    #[test]
    fn test_mh_prior_uniform() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            let u = Uniform::new(0.0, 1.0);
            r.sample(u)
        }

        let mut rng = rand::thread_rng();
        let x = mh_prior(0.0, loglike, prior_draw, 25, &mut rng);

        assert!(x <= 1.0);
        assert!(x >= 0.0);
    }

    #[test]
    fn test_mh_prior_normal() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            let norm = Normal::new(0.0, 1.0);
            r.sample(norm)
        }

        let mut rng = rand::thread_rng();

        // Smoke test. just make sure nothing blows up
        let _x = mh_prior(0.0, loglike, prior_draw, 25, &mut rng);
    }
}

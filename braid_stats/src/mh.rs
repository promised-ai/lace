use rand::Rng;
use std::f64;

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
    let x = x_start;
    let fx = loglike(&x);
    (0..n_iters)
        .fold((x, fx), |(x, fx), _| {
            let y = prior_draw(&mut rng);
            let fy = loglike(&y);
            let r: f64 = rng.gen::<f64>();
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .0
}

/// Symmetric random walk MCMC
///
/// # Arguments
/// - x_start: the starting value
/// - score_fn: the score function. For Bayesian inference: f(x|θ)π(θ)
/// - walk_fn: a symmetric transition function q(x -> x') = q(x' -> x). Should
///   enforce the domain bounds.
/// - n_iters: the number of MH steps
/// - rng: The random number generator
pub fn mh_symrw<T, F, Q, R>(
    x_start: T,
    score_fn: F,
    walk_fn: Q,
    n_iters: usize,
    mut rng: &mut R,
) -> T
where
    F: Fn(&T) -> f64,
    Q: Fn(&T, &mut R) -> T,
    R: Rng,
{
    let score_x = score_fn(&x_start);
    let x = x_start;
    (0..n_iters)
        .fold((x, score_x), |(x, score_x), _| {
            let y = walk_fn(&x, &mut rng);
            let score_y = score_fn(&y);
            let r: f64 = rng.gen::<f64>();
            if r.ln() < score_y - score_x {
                (y, score_y)
            } else {
                (x, score_x)
            }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Normal;

    #[test]
    fn test_mh_prior_uniform() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            r.gen()
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

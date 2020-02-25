use rand::Rng;
use std::f64;

/// Information from the last step of a Metropolis-Hastings (MH) update
pub struct MhResult<T> {
    /// The final value of the Markov chain
    pub x: T,
    /// The final score value of x. This function will depend on what type of
    /// sampler is being used.
    pub score_x: f64,
}

impl<T> From<(T, f64)> for MhResult<T> {
    fn from(tuple: (T, f64)) -> MhResult<T> {
        MhResult {
            x: tuple.0,
            score_x: tuple.1,
        }
    }
}

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
) -> MhResult<T>
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
        .into()
}

/// Draw posterior samples from f(x|y)π(x) by taking proposals from a static
/// importance distribution, Q.
///
/// # Arguments
/// - x_start: the starting value
/// - ln_f: The log proportional posterior
/// - q_draw: Function that takes a Rng and draws from Q.
/// - q_ln_f: Function that evaluates the log likelihood of Q at x
/// - n_iters: the number of MH iterations
/// - rng: The random number generator
pub fn mh_importance<T, Fx, Dq, Fq, R: Rng>(
    x_start: T,
    ln_f: Fx,
    q_draw: Dq,
    q_ln_f: Fq,
    n_iters: usize,
    mut rng: &mut R,
) -> MhResult<T>
where
    Fx: Fn(&T) -> f64,
    Dq: Fn(&mut R) -> T,
    Fq: Fn(&T) -> f64,
{
    let x = x_start;
    let fx = ln_f(&x) - q_ln_f(&x);
    (0..n_iters)
        .fold((x, fx), |(x, fx), _| {
            let y = q_draw(&mut rng);
            let fy = ln_f(&y) - q_ln_f(&y);
            let r: f64 = rng.gen::<f64>();
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .into()
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
) -> MhResult<T>
where
    F: Fn(&T) -> f64,
    Q: Fn(&T, &mut R) -> T,
    R: Rng,
{
    let score_x = score_fn(&x_start);
    let x = x_start;
    (0..n_iters)
        .fold((x, score_x), |(x, fx), _| {
            let y = walk_fn(&x, &mut rng);
            let fy = score_fn(&y);
            let r: f64 = rng.gen::<f64>();
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Normal;
    use rv::dist::{Bernoulli, Beta, Gaussian};
    use rv::misc::ks_test;
    use rv::traits::{Cdf, Rv};

    const KS_PVAL: f64 = 0.2;
    const N_FLAKY_TEST: usize = 10;

    fn mh_chain<F, R>(
        x_start: f64,
        mh_fn: F,
        n_steps: usize,
        mut rng: &mut R,
    ) -> Vec<f64>
    where
        F: Fn(&f64, &mut R) -> f64,
        R: Rng,
    {
        let mut x = x_start;
        let mut samples = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let y = mh_fn(&x, &mut rng);
            samples.push(y);
            x = y
        }

        samples
    }

    #[test]
    fn test_mh_prior_uniform() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            r.gen()
        }

        let mut rng = rand::thread_rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                0.5,
                |&x, mut rng| mh_prior(x, loglike, prior_draw, 1, &mut rng).x,
                500,
                &mut rng,
            );
            let (_, p) = ks_test(&xs, |x| x);

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_mh_prior_gaussian() {
        let gauss = Gaussian::standard();
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            let norm = Normal::new(0.0, 1.0).unwrap();
            r.sample(norm)
        }

        let mut rng = rand::thread_rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                0.5,
                |&x, mut rng| mh_prior(x, loglike, prior_draw, 1, &mut rng).x,
                500,
                &mut rng,
            );
            let (_, p) = ks_test(&xs, |x| gauss.cdf(&x));

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_mh_importance_beta() {
        let xs: Vec<u8> = vec![0, 0, 1, 1, 1, 1];
        let prior = Beta::new(2.0, 2.0).unwrap();

        // Proportional to the posterior
        let ln_fn = |theta: &f64| {
            let likelihood = Bernoulli::new(*theta).unwrap();
            let f: f64 = xs.iter().map(|x| likelihood.ln_f(x)).sum();
            f + prior.ln_f(theta)
        };

        fn q_draw<R: Rng>(mut rng: &mut R) -> f64 {
            let q = Beta::new(2.0, 1.0).unwrap();
            q.draw(&mut rng)
        }

        fn q_ln_f(theta: &f64) -> f64 {
            let q = Beta::new(2.0, 1.0).unwrap();
            q.ln_f(theta)
        }

        let mut rng = rand::thread_rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                0.5,
                |&x, mut rng| {
                    mh_importance(x, ln_fn, q_draw, q_ln_f, 2, &mut rng).x
                },
                250,
                &mut rng,
            );

            let true_posterior = Beta::new(2.0 + 4.0, 2.0 + 2.0).unwrap();

            let (ks, p) = ks_test(&xs, |x| true_posterior.cdf(&x));

            println!("KS {}, P {}", ks, p);

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_symrw_uniform() {
        let score_fn = |_x: &f64| 0.0;
        fn walk_fn<R: Rng>(x: &f64, r: &mut R) -> f64 {
            let norm = Normal::new(*x, 0.2).unwrap();
            let y = r.sample(norm).rem_euclid(1.0);
            y
        }

        let mut rng = rand::thread_rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                0.5,
                |&x, mut rng| mh_symrw(x, score_fn, walk_fn, 1, &mut rng).x,
                500,
                &mut rng,
            );
            let (_, p) = ks_test(&xs, |x| x);

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_symrw_gaussian() {
        let gauss = Gaussian::new(1.0, 1.5).unwrap();

        let score_fn = |x: &f64| gauss.ln_f(x);
        fn walk_fn<R: Rng>(x: &f64, r: &mut R) -> f64 {
            let norm = Normal::new(*x, 0.5).unwrap();
            r.sample(norm)
        }

        let mut rng = rand::thread_rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                1.0,
                |&x, mut rng| mh_symrw(x, score_fn, walk_fn, 10, &mut rng).x,
                250,
                &mut rng,
            );
            let (_, p) = ks_test(&xs, |x| gauss.cdf(&x));
            println!("p: {}", p);

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }
}

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
    rng: &mut R,
) -> MhResult<T>
where
    F: Fn(&T) -> f64,
    D: Fn(&mut R) -> T,
{
    let x = x_start;
    let fx = loglike(&x);
    (0..n_iters)
        .fold((x, fx), |(x, fx), _| {
            let y = prior_draw(rng);
            let fy = loglike(&y);

            assert!(fy.is_finite(), "Non finite proposal likelihood");

            let r: f64 = rng.random::<f64>();
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .into()
}

// TODO: rename this to mc_importance, because importance in Monte Carlo, not
// Metropolis-Hastings
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
    rng: &mut R,
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
            let y = q_draw(rng);
            let fy = ln_f(&y) - q_ln_f(&y);

            assert!(fy.is_finite(), "Non finite proposal likelihood");

            let r: f64 = rng.random::<f64>();
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
    rng: &mut R,
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
            let y = walk_fn(&x, rng);
            let fy = score_fn(&y);

            assert!(fy.is_finite(), "Non finite proposal likelihood");

            let r: f64 = rng.random::<f64>();
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .into()
}

fn slice_stepping_out<F>(
    ln_height: f64,
    x: f64,
    step_size: f64,
    score_fn: &F,
    r: f64,
    bounds: (f64, f64),
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let step_limit = 15_usize;

    let x_left = {
        let mut x_left = r.mul_add(-step_size, x);
        let mut loop_counter: usize = 0;
        let mut step = step_size;
        loop {
            let ln_fx_left = score_fn(x_left);
            if x_left < bounds.0 {
                break bounds.0;
            } else if ln_fx_left < ln_height {
                break x_left;
            }

            x_left -= step;
            step *= 2.0;

            if loop_counter == step_limit {
                panic!(
                    "x_left step ({}/{}) limit ({}) hit. x = {}, height = {}, fx = {}",
                    step_size, step, step_limit, x, ln_height, ln_fx_left,
                )
            }
            loop_counter += 1;
        }
    };

    let x_right = {
        let mut x_right = (1.0 - r).mul_add(step_size, x);
        let mut loop_counter: usize = 0;
        let mut step = step_size;
        loop {
            let ln_fx_right = score_fn(x_right);
            if x_right > bounds.1 {
                break bounds.1;
            } else if ln_fx_right < ln_height {
                break x_right;
            }

            x_right += step;
            step *= 2.0;

            if loop_counter == step_limit {
                panic!("x_right step limit ({}) hit", step_limit)
            }
            loop_counter += 1;
        }
    };

    (x_left, x_right)
}

fn mh_slice_step<F, R>(
    x_start: f64,
    step_size: f64,
    score_fn: &F,
    bounds: (f64, f64),
    rng: &mut R,
) -> MhResult<f64>
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    use crate::rv::dist::Uniform;
    use crate::rv::traits::Sampleable;

    let ln_fx = score_fn(x_start);
    let ln_u = rng.random::<f64>().ln() + ln_fx;
    let (mut x_left, mut x_right) = slice_stepping_out(
        ln_u,
        x_start,
        step_size,
        &score_fn,
        rng.random::<f64>(),
        bounds,
    );

    let step_limit = 50;
    let mut loop_counter = 0;
    loop {
        let x: f64 = Uniform::new_unchecked(x_left, x_right).draw(rng);
        let ln_fx = score_fn(x);
        // println!("{}: ({}, {}) - [{}, {}]", x, x_left, x_right, ln_u, ln_fx);
        if ln_fx > ln_u {
            break MhResult { x, score_x: ln_fx };
        }

        if loop_counter == step_limit {
            panic!("Slice interval tuning limit ({}) hit", step_limit)
        }

        if x > x_start {
            x_right = x;
        } else {
            x_left = x;
        };

        loop_counter += 1;
    }
}

/// Uses a slice sampler w/ the stepping out method to draw from a univariate
/// posterior distribution.
///
/// # Notes
/// Under some circumstances, the stepping out will hit the max iterations and
/// cause a panic. You might want to stay away from this sampler if you don't
/// know that your posterior is well behaved.
pub fn mh_slice<F, R>(
    x_start: f64,
    step_size: f64,
    n_iters: usize,
    score_fn: F,
    bounds: (f64, f64),
    rng: &mut R,
) -> MhResult<f64>
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    (0..n_iters).fold(
        mh_slice_step(x_start, step_size, &score_fn, bounds, rng),
        |acc, _| mh_slice_step(acc.x, step_size, &score_fn, bounds, rng),
    )
}

pub fn mh_symrw_adaptive<F, R>(
    x_start: f64,
    mut mu_guess: f64,
    mut var_guess: f64,
    n_steps: usize,
    score_fn: F,
    bounds: (f64, f64),
    rng: &mut R,
) -> MhResult<f64>
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    use crate::rv::dist::Gaussian;
    use crate::rv::traits::Sampleable;

    // FIXME: initialize this properly
    let gamma_init = 0.9;

    let mut x = x_start;
    let mut fx = score_fn(x);
    let mut x_sum = x;
    let lambda: f64 = 2.38 * 2.38;

    for n in 0..n_steps {
        let y: f64 =
            Gaussian::new_unchecked(x, (lambda * var_guess).sqrt()).draw(rng);
        if bounds.0 < x || x < bounds.1 {
            let fy = score_fn(y);

            assert!(fy.is_finite(), "Non finite proposal likelihood");

            if rng.random::<f64>().ln() < fy - fx {
                x = y;
                fx = fy;
            }
        }
        x_sum += x;
        let x_bar = x_sum / (n + 1) as f64;
        let gamma = gamma_init / (n + 1) as f64;
        let mu_next = (x_bar - mu_guess).mul_add(gamma, mu_guess);
        var_guess = (x - mu_guess)
            .mul_add(x - mu_guess, -var_guess)
            .mul_add(gamma, var_guess);
        mu_guess = mu_next;
    }

    // println!("[A: {}], (mu, sigma) = ({}, {})", acc / n_steps as f64, mu_guess, var_guess.sqrt());

    MhResult { x, score_x: fx }
}

use crate::mat::{MeanVector, ScaleMatrix, SquareT};
use std::ops::Mul;

/// Multivariate adaptive Metropolis-Hastings sampler using globally adaptive
/// symmetric random walk.
///
/// # Notes
///
/// This sampler is slow and unstable due to all the matrix math, and often does
/// not achieve the correct stationary distribution, but most often achieves the
/// correct posterior mean -- if you care about that.
pub fn mh_symrw_adaptive_mv<F, R, M, S>(
    x_start: M,
    mut mu_guess: M,
    mut var_guess: S,
    n_steps: usize,
    score_fn: F,
    bounds: &[(f64, f64)],
    rng: &mut R,
) -> MhResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
    R: Rng,
    M: MeanVector + SquareT<Output = S> + Mul<f64, Output = M>,
    S: ScaleMatrix + Mul<f64, Output = S>,
{
    use crate::rv::dist::MvGaussian;
    use crate::rv::nalgebra::{DMatrix, DVector};
    use crate::rv::traits::Sampleable;

    // TODO: initialize this properly
    // let gamma = (n_steps as f64).recip();
    let gamma = 0.5;

    let mut x = x_start;
    let mut fx = score_fn(x.values());
    let mut x_sum = M::zeros().mv_add(&x);
    let mut ln_lambda: f64 = (2.38 * 2.38 / x.len() as f64).ln();

    let n_rows = x.len();

    for n in 0..n_steps {
        var_guess.diagonalize();
        let cov = DMatrix::from_row_slice(n_rows, n_rows, var_guess.values());
        let mu = DVector::from_row_slice(x.values());

        let y: DVector<f64> =
            MvGaussian::new_unchecked(mu, ln_lambda.exp() * cov).draw(rng);
        let y = M::from_dvector(y);

        let in_bounds = y
            .values()
            .iter()
            .zip(bounds.iter())
            .all(|(&y_i, bounds_i)| bounds_i.0 < y_i && y_i < bounds_i.1);

        let alpha = if in_bounds {
            let fy = score_fn(y.values());

            assert!(fy.is_finite(), "Non finite proposal likelihood");

            let ln_alpha = (fy - fx).min(0.0);
            if rng.random::<f64>().ln() < ln_alpha {
                x = y;
                fx = fy;
            }
            ln_alpha.exp()
        } else {
            0.0
        };

        x_sum = x_sum.mv_add(&x);
        ln_lambda += gamma * (alpha - 0.234);

        let x_bar = M::zeros().mv_add(&x_sum) * (n as f64 + 1.0).recip();
        let mu_next = (x_bar.mv_sub(&mu_guess) * gamma).mv_add(&mu_guess);
        var_guess = (x.clone().mv_sub(&mu_guess).square_t().mv_sub(&var_guess)
            * gamma)
            .mv_add(&var_guess);
        mu_guess = mu_next;
    }

    MhResult {
        x: Vec::from(x.values()),
        score_x: fx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rv::dist::{Bernoulli, Beta, Gaussian};
    use crate::rv::misc::ks_test;
    use crate::rv::traits::{Cdf, HasDensity, Sampleable};
    use rand_distr::Normal;

    const KS_PVAL: f64 = 0.2;
    const N_FLAKY_TEST: usize = 10;

    fn mh_chain<F, X, R>(
        x_start: X,
        mh_fn: F,
        n_steps: usize,
        rng: &mut R,
    ) -> Vec<X>
    where
        X: Clone,
        F: Fn(&X, &mut R) -> X,
        R: Rng,
    {
        let mut x = x_start;
        let mut samples: Vec<X> = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let y = mh_fn(&x, rng);
            samples.push(y.clone());
            x = y
        }

        samples
    }

    #[test]
    fn test_mh_prior_uniform() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            r.random()
        }

        let mut rng = rand::rng();
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

        let mut rng = rand::rng();
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

        let mut rng = rand::rng();
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

            let (_, p) = ks_test(&xs, |x| true_posterior.cdf(&x));

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

            r.sample(norm).rem_euclid(1.0)
        }

        let mut rng = rand::rng();
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

        let mut rng = rand::rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                1.0,
                |&x, mut rng| mh_symrw(x, score_fn, walk_fn, 10, &mut rng).x,
                250,
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
    fn test_mh_slice_uniform() {
        let loglike = |x: f64| {
            if 0.0 < x && x < 1.0 {
                0.0
            } else {
                std::f64::NEG_INFINITY
            }
        };

        let mut rng = rand::rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                0.5,
                |&x, mut rng| {
                    mh_slice(x, 0.2, 1, loglike, (0.0, 1.0), &mut rng).x
                },
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
    fn test_mh_slice_gaussian() {
        use std::f64::{INFINITY, NEG_INFINITY};

        let gauss = Gaussian::new(1.0, 1.5).unwrap();

        let score_fn = |x: f64| gauss.ln_f(&x);

        let mut rng = rand::rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                1.0,
                |&x, mut rng| {
                    mh_slice(
                        x,
                        1.0,
                        1,
                        score_fn,
                        (NEG_INFINITY, INFINITY),
                        &mut rng,
                    )
                    .x
                },
                250,
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
    fn test_mh_symrw_adaptive_gaussian() {
        use std::f64::{INFINITY, NEG_INFINITY};

        let gauss = Gaussian::new(1.0, 1.5).unwrap();

        let score_fn = |x: f64| gauss.ln_f(&x);

        let mut rng = rand::rng();
        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let xs = mh_chain(
                1.0,
                |&x, mut rng| {
                    mh_symrw_adaptive(
                        x,
                        0.1,
                        0.1,
                        100,
                        score_fn,
                        (NEG_INFINITY, INFINITY),
                        &mut rng,
                    )
                    .x
                },
                250,
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
    fn test_mh_symrw_adaptive_normal_gamma() {
        use std::f64::{INFINITY, NEG_INFINITY};

        let mut rng = rand::rng();
        let sigma: f64 = 1.5;
        let m0: f64 = 0.0;
        let s0: f64 = 0.5;
        let gauss = Gaussian::new(1.0, sigma).unwrap();
        let prior = Gaussian::new(m0, s0).unwrap();

        let xs: Vec<f64> = gauss.sample(20, &mut rng);
        let sum_x = xs.iter().sum::<f64>();

        let score_fn = |mu: f64| {
            let g = Gaussian::new_unchecked(mu, sigma);
            let fx: f64 = xs.iter().map(|x| g.ln_f(x)).sum();
            fx + prior.ln_f(&mu)
        };

        let posterior = {
            let nf = xs.len() as f64;
            let s2 = sigma * sigma;
            let s02 = s0 * s0;
            let sn = ((nf / s2) + s02.recip()).recip();
            let mn = sn * (m0 / s02 + sum_x / s2);
            Gaussian::new(mn, sn.sqrt()).unwrap()
        };

        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let ys = mh_chain(
                1.0,
                |&x, mut rng| {
                    mh_symrw_adaptive(
                        x,
                        0.1,
                        0.1,
                        100,
                        score_fn,
                        (NEG_INFINITY, INFINITY),
                        &mut rng,
                    )
                    .x
                },
                250,
                &mut rng,
            );
            let (_, p) = ks_test(&ys, |y| posterior.cdf(&y));

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_mh_symrw_mv_normal_gamma_known_var() {
        let mut rng = rand::rng();
        let sigma: f64 = 1.5;
        let m0: f64 = 0.0;
        let s0: f64 = 0.5;
        let gauss = Gaussian::new(1.0, sigma).unwrap();
        let prior = Gaussian::new(m0, s0).unwrap();

        let xs: Vec<f64> = gauss.sample(20, &mut rng);
        let sum_x = xs.iter().sum::<f64>();

        let score_fn = |mu: &f64| {
            let g = Gaussian::new_unchecked(*mu, sigma);
            let fx: f64 = xs.iter().map(|x| g.ln_f(x)).sum();
            fx + prior.ln_f(mu)
        };

        fn walk_fn<R: Rng>(x: &f64, r: &mut R) -> f64 {
            Gaussian::new_unchecked(*x, 0.2).draw(r)
        }

        let posterior = {
            let nf = xs.len() as f64;
            let s2 = sigma * sigma;
            let s02 = s0 * s0;

            let sn = ((nf / s2) + s02.recip()).recip();

            let mn = sn * (m0 / s02 + sum_x / s2);
            println!("Posterior mean: {}", mn);

            Gaussian::new(mn, sn.sqrt()).unwrap()
        };

        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let ys = mh_chain(
                1.0,
                |x, mut rng| mh_symrw(*x, score_fn, walk_fn, 50, &mut rng).x,
                250,
                &mut rng,
            );
            let (_, p) = ks_test(&ys, |y| posterior.cdf(&y));
            println!("p: {}, m: {}", p, lace_utils::mean(&ys));

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_mh_symrw_adaptive_mv_normal_gamma_known_var() {
        use crate::mat::Matrix1x1;
        use std::f64::{INFINITY, NEG_INFINITY};

        let mut rng = rand::rng();
        let sigma: f64 = 1.5;
        let m0: f64 = 0.0;
        let s0: f64 = 0.5;
        let gauss = Gaussian::new(1.0, sigma).unwrap();
        let prior = Gaussian::new(m0, s0).unwrap();

        let xs: Vec<f64> = gauss.sample(20, &mut rng);
        let sum_x = xs.iter().sum::<f64>();

        let score_fn = |mu: &[f64]| {
            let g = Gaussian::new_unchecked(mu[0], sigma);
            let fx: f64 = xs.iter().map(|x| g.ln_f(x)).sum();
            fx + prior.ln_f(&mu[0])
        };

        let posterior = {
            let nf = xs.len() as f64;
            let s2 = sigma * sigma;
            let s02 = s0 * s0;

            let sn = ((nf / s2) + s02.recip()).recip();

            let mn = sn * (m0 / s02 + sum_x / s2);
            println!("Posterior mean: {}", mn);

            Gaussian::new(mn, sn.sqrt()).unwrap()
        };

        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let ys = mh_chain(
                1.0,
                |x, mut rng| {
                    mh_symrw_adaptive_mv(
                        Matrix1x1([*x]),
                        Matrix1x1([0.1]),
                        Matrix1x1([0.1]),
                        10,
                        score_fn,
                        &[(NEG_INFINITY, INFINITY)],
                        &mut rng,
                    )
                    .x[0]
                },
                250,
                &mut rng,
            );
            let (_, p) = ks_test(&ys, |y| posterior.cdf(&y));
            println!("p: {}, m: {}", p, lace_utils::mean(&ys));

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }

    #[test]
    fn test_mh_symrw_adaptive_mv_normal_gamma_unknown() {
        use crate::mat::{Matrix2x2, Vector2};
        use crate::rv::dist::InvGamma;
        use crate::test::gauss_perm_test;
        use std::f64::{INFINITY, NEG_INFINITY};

        let n = 20;
        let n_samples = 250;

        let mut rng = rand::rng();

        // Prior parameters
        let m0: f64 = 0.0;
        let v0: f64 = 0.5;
        let a0: f64 = 1.5;
        let b0: f64 = 1.0;

        // True distribution
        let gauss = Gaussian::new(1.0, 1.5).unwrap();

        // prior on sigma
        let prior_var = InvGamma::new(a0, b0).unwrap();

        // Generate data and get sufficient statistics
        let xs: Vec<f64> = gauss.sample(n, &mut rng);
        let sum_x = xs.iter().sum::<f64>();
        let sum_x_sq = xs.iter().map(|&x| x * x).sum::<f64>();

        println!("Mean(x): {}", sum_x / n as f64);

        // The proportional posterior for MCMC
        let score_fn = |mu_var: &[f64]| {
            let mu = mu_var[0];
            let var = mu_var[1];
            let sigma = var.sqrt();
            let g = Gaussian::new_unchecked(mu, sigma);
            let fx: f64 = xs.iter().map(|x| g.ln_f(x)).sum();
            let prior_mu = Gaussian::new(m0, v0.sqrt() * sigma).unwrap();
            fx + prior_mu.ln_f(&mu) + prior_var.ln_f(&var)
        };

        // Compute the normal inverse-gamma posterior according kevin murphy's
        // whitepaper
        let posterior_samples: Vec<(f64, f64)> = {
            let nf = n as f64;

            let v0_inv = v0.recip();
            let vn_inv = v0_inv + nf;
            let mn_over_vn = v0_inv.mul_add(m0, sum_x);
            let mn = mn_over_vn * vn_inv.recip();
            let an = a0 + nf / 2.0;
            let bn = 0.5_f64.mul_add(
                (mn * mn).mul_add(-vn_inv, (m0 * m0).mul_add(v0_inv, sum_x_sq)),
                b0,
            );
            let vn_sqrt = vn_inv.recip().sqrt();

            let post_var = InvGamma::new(an, bn).unwrap();
            (0..n_samples)
                .map(|_| {
                    let var: f64 = post_var.draw(&mut rng);
                    let mu: f64 = Gaussian::new(mn, vn_sqrt * var.sqrt())
                        .unwrap()
                        .draw(&mut rng);
                    (mu, var)
                })
                .collect()
        };

        let (mean_mu, var_mu) = {
            use lace_utils::{mean, var};
            let mus: Vec<f64> =
                posterior_samples.iter().map(|xy| xy.0).collect();
            (mean(&mus), var(&mus))
        };
        println!("Posterior Mean/Var: {}/{}", mean_mu, var_mu);

        let n_passes = (0..N_FLAKY_TEST).fold(0, |acc, _| {
            let mcmc_samples: Vec<(f64, f64)> = mh_chain(
                (1.0, 1.0),
                |x, mut rng| {
                    let x = mh_symrw_adaptive_mv(
                        Vector2([x.0, x.1]),
                        Vector2([0.1, 1.0]),
                        Matrix2x2::from_diag([1.0, 1.0]),
                        100,
                        score_fn,
                        &[(NEG_INFINITY, INFINITY), (0.0, INFINITY)],
                        &mut rng,
                    )
                    .x;
                    (x[0], x[1])
                },
                n_samples,
                &mut rng,
            );

            let (mean_mu_mh, var_mu_mh) = {
                use lace_utils::{mean, var};
                let mus: Vec<f64> =
                    mcmc_samples.iter().map(|xy| xy.0).collect();
                (mean(&mus), var(&mus))
            };

            let p = gauss_perm_test(
                posterior_samples.clone(),
                mcmc_samples,
                500,
                &mut rng,
            );

            println!("p: {}", p);
            println!("MCMC Mean/Var: {}/{}", mean_mu_mh, var_mu_mh);

            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(n_passes > 0);
    }
}

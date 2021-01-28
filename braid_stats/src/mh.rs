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
        let mut x_left = x - r * step_size;
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
        let mut x_right = x + (1.0 - r) * step_size;
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
    mut rng: &mut R,
) -> MhResult<f64>
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    use rv::dist::Uniform;
    use rv::traits::Rv;

    let ln_fx = score_fn(x_start);
    let ln_u = rng.gen::<f64>().ln() + ln_fx;
    let (mut x_left, mut x_right) = slice_stepping_out(
        ln_u,
        x_start,
        step_size,
        &score_fn,
        rng.gen::<f64>(),
        bounds,
    );

    let step_limit = 50;
    let mut loop_counter = 0;
    loop {
        let x: f64 = Uniform::new_unchecked(x_left, x_right).draw(&mut rng);
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

pub fn mh_slice<F, R>(
    x_start: f64,
    step_size: f64,
    n_iters: usize,
    score_fn: F,
    bounds: (f64, f64),
    mut rng: &mut R,
) -> MhResult<f64>
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    (0..n_iters).fold(
        mh_slice_step(x_start, step_size, &score_fn, bounds.clone(), &mut rng),
        |acc, _| {
            mh_slice_step(acc.x, step_size, &score_fn, bounds.clone(), &mut rng)
        },
    )
}

pub fn mh_symrw_adaptive<F, R>(
    x_start: f64,
    mut mu_guess: f64,
    mut var_guess: f64,
    n_steps: usize,
    score_fn: F,
    bounds: (f64, f64),
    mut rng: &mut R,
) -> MhResult<f64>
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    use rv::dist::Gaussian;
    use rv::traits::Rv;

    // FIXME: initialize this properly
    let gamma = 0.9;

    let mut x = x_start;
    let mut fx = score_fn(x);
    let mut x_sum = x;
    // let mut acc = 0.0;
    let lambda: f64 = 2.38 * 2.38;

    for n in 0..n_steps {
        let y: f64 = Gaussian::new_unchecked(x, (lambda * var_guess).sqrt())
            .draw(&mut rng);
        if bounds.0 < x || x < bounds.1 {
            let fy = score_fn(y);
            if rng.gen::<f64>().ln() < fy - fx {
                // acc += 1.0;
                x = y;
                fx = fy;
            }
        }
        x_sum += x;
        let x_bar = x_sum / (n + 1) as f64;
        let mu_next = mu_guess + gamma * (x_bar - mu_guess);
        var_guess = var_guess + gamma * ((x - mu_guess).powi(2) - var_guess);
        mu_guess = mu_next;
    }

    // println!("[A: {}], (mu, sigma) = ({}, {})", acc / n_steps as f64, mu_guess, var_guess.sqrt());

    MhResult { x, score_x: fx }
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

        let mut rng = rand::thread_rng();
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

        let mut rng = rand::thread_rng();
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

        let mut rng = rand::thread_rng();
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

        let mut rng = rand::thread_rng();
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
}

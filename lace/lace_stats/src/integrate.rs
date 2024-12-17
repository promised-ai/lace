use lace_consts::rv::misc::LogSumExp;
use rand::Rng;

/// Monte Carlo integration
///
/// # Arguments
///
/// - ln_f: the log of the function to integrate with the draw term adjusted
///   for. For example, if you were estimating the marginal likelihood, `ln_f`
///   would be the log likelihood and `draw` would draw from the prior.
/// - draw: A function that draws samples to evaluate in `ln_f`
/// - n_iters: the number of samples to use for estimation
/// - rng: A random number generator
pub fn mc_integral<X, Fx, D, R>(
    ln_f: Fx,
    draw: D,
    n_iters: usize,
    rng: &mut R,
) -> f64
where
    Fx: Fn(&X) -> f64,
    D: Fn(&mut R) -> X,
    R: Rng,
{
    // NOTE: computing the max value for logsumexp in the map saves a
    // statistically insignificant amount of time and makes the code a lot
    // longer.
    let logsumexp = (0..n_iters).map(|_| ln_f(&draw(rng))).logsumexp();

    logsumexp - (n_iters as f64).ln()
}

/// Importance Sampling integration
///
/// # Arguments
///
/// - ln_f: the log of the function to integrate.
/// - q_draw: A function that draws samples from the importance distribution
/// - q_ln_f: A function that draws samples from the importance distribution
/// - n_iters: the number of samples to use for estimation
/// - rng: A random number generator
pub fn importance_integral<X, Fx, Dq, Fq, R>(
    ln_f: Fx,
    q_draw: Dq,
    q_ln_f: Fq,
    n_iters: usize,
    rng: &mut R,
) -> f64
where
    Fx: Fn(&X) -> f64,
    Dq: Fn(&mut R) -> X,
    Fq: Fn(&X) -> f64,
    R: Rng,
{
    let logsumexp = (0..n_iters)
        .map(|_| {
            let x: X = q_draw(rng);
            ln_f(&x) - q_ln_f(&x)
        })
        .logsumexp();

    logsumexp - (n_iters as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rv::data::DataOrSuffStat;
    use crate::rv::dist::{Bernoulli, Beta};
    use crate::rv::traits::{ConjugatePrior, HasDensity, Sampleable};

    fn relerr(x: f64, x_est: f64) -> f64 {
        (x_est / x - 1.0).abs()
    }

    #[test]
    fn bb_log_marginal_mh() {
        let mut rng = rand::thread_rng();

        let xs: Vec<u8> = vec![0, 0, 1, 1, 1, 1];

        let ln_f = |theta: &f64| {
            let likelihood = Bernoulli::new(*theta).unwrap();
            let f: f64 = xs.iter().map(|x| likelihood.ln_f(x)).sum();
            f
        };

        fn draw<R: Rng>(mut rng: &mut R) -> f64 {
            let prior = Beta::new(2.0, 2.0).unwrap();
            prior.draw(&mut rng)
        }

        let n_passes = (0..5).fold(0, |acc, _| {
            let est = mc_integral(ln_f, draw, 100_000, &mut rng);
            let truth = {
                let prior = Beta::new(2.0, 2.0).unwrap();
                let data = DataOrSuffStat::Data(&xs);
                prior.ln_m(&data)
            };
            let err = relerr(truth, est);

            if err > 1e-3 {
                acc
            } else {
                acc + 1
            }
        });

        assert!(n_passes > 2);
    }

    #[test]
    fn bb_log_marginal_importance() {
        let mut rng = rand::thread_rng();

        let xs: Vec<u8> = vec![0, 0, 1, 1, 1, 1];
        let prior = Beta::new(2.0, 2.0).unwrap();

        let ln_f = |theta: &f64| {
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

        let n_passes = (0..5).fold(0, |acc, _| {
            let est =
                importance_integral(ln_f, q_draw, q_ln_f, 100_000, &mut rng);
            let truth = {
                let data = DataOrSuffStat::Data(&xs);
                prior.ln_m(&data)
            };
            let err = relerr(truth, est);

            if err > 1e-3 {
                acc
            } else {
                acc + 1
            }
        });

        assert!(n_passes > 2);
    }
}

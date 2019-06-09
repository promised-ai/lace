use braid_utils::misc::logsumexp;
use rand::Rng;

/// Monte Carlo integration
///
/// # Aguments
///
/// - ln_f: the log of the function to integrate with the draw term adjusted
///   for. For example, if you were estimating the marginal likelihood, `ln_f`
///   would be the log likelihood and `draw` would draw from the prior.
/// - draw: A function that draws samples to evaluate in `ln_f`
/// - n_iters: the number of samples to use for estimation
/// - rng: A random number generator
pub fn mc_integral<H, L, D, R>(
    ln_f: L,
    draw: D,
    n_iters: usize,
    mut rng: &mut R,
) -> f64
where
    L: Fn(H) -> f64,
    D: Fn(&mut R) -> H,
    R: Rng,
{
    let loglikes: Vec<f64> =
        (0..n_iters).map(|_| ln_f(draw(&mut rng))).collect();

    logsumexp(&loglikes) - (n_iters as f64).ln()
}

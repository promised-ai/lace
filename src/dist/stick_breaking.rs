extern crate rand;
extern crate rv;

use self::rand::Rng;
use self::rv::dist::Beta;
use self::rv::traits::Rv;
use std::io;

const MAX_STICK_BREAKING_ITERS: u64 = 1000;

/// Append new dirchlet weights by stick breaking until the new weight is less
/// than u*
pub fn sb_slice_extend<R: Rng>(
    mut weights: Vec<f64>,
    alpha: f64,
    u_star: f64,
    mut rng: &mut R,
) -> io::Result<Vec<f64>> {
    let mut b_star = weights.pop().unwrap();

    // If α is low and we do the dirichlet update w ~ Dir(n_1, ..., n_k, α),
    // the final weight wil oven be zero. In that case, we're done.
    if b_star <= 1E-16 {
        weights.push(b_star);
        return Ok(weights);
    }

    let beta = Beta::new(1.0, alpha)?;

    let mut iters: u64 = 0;
    loop {
        let vk: f64 = beta.draw(&mut rng);
        let bk = vk * b_star;
        b_star *= 1.0 - vk;

        weights.push(bk);

        if b_star < u_star {
            if b_star > 0.0 {
                weights.push(b_star);
            }
            if weights.iter().any(|&w| w <= 0.0) {
                let err_kind = io::ErrorKind::InvalidData;
                return Err(io::Error::new(err_kind, "Invalid weights"));
            }
            return Ok(weights);
        }

        iters += 1;
        if iters > MAX_STICK_BREAKING_ITERS {
            let err_kind = io::ErrorKind::TimedOut;
            return Err(io::Error::new(err_kind, "Max iters reached"));
        }
    }
}

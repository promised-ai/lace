extern crate rand;
extern crate rv;

use self::rand::Rng;
use self::rv::dist::{Beta, Dirichlet};
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
    let beta = Beta::new(1.0, alpha)?;
    let k = weights.len();
    let weights_start = weights.clone();

    let mut iters: u64 = 0;
    loop {
        let vk: f64 = beta.draw(&mut rng);
        let bk = vk * b_star;
        b_star *= 1.0 - vk;

        weights.push(bk);

        if b_star < u_star {
            weights.push(b_star);
            assert!(weights.iter().all(|&w| w > 0.0));
            return Ok(weights);
        }

        iters += 1;
        if iters > MAX_STICK_BREAKING_ITERS {
            println!("weights: {:?}", weights_start);
            println!("α:  {:?}", alpha);
            println!("u*: {:?}", u_star);
            println!("k:  {:?}", k);
            let err_kind = io::ErrorKind::TimedOut;
            return Err(io::Error::new(err_kind, "Max iters reached"));
        }
    }
}

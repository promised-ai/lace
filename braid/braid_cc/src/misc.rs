use std::ops::Index;

use braid_utils::Shape;
use rand::Rng;
use rv::dist::Beta;
use rv::misc::pflip;
use rv::traits::Rv;
use serde::{Deserialize, Serialize};

const MAX_STICK_BREAKING_ITERS: u16 = 1000;

/// Draw n categorical indices in {0,..,k-1} from an n-by-k vector of vectors
/// of un-normalized log probabilities.
///
/// Automatically chooses whether to use serial or parallel computing.
pub fn massflip<M>(logps: M, mut rng: &mut impl Rng) -> Vec<usize>
where
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    braid_flippers::massflip_mat_par(logps, &mut rng)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct CrpDraw {
    pub asgn: Vec<usize>,
    pub counts: Vec<usize>,
    pub n_cats: usize,
}

/// Draw from Chinese Restaraunt Process
pub fn crp_draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> CrpDraw {
    let mut n_cats = 0;
    let mut weights: Vec<f64> = vec![];
    let mut asgn: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..n {
        weights.push(alpha);
        let k = pflip(&weights, 1, rng)[0];
        asgn.push(k);

        if k == n_cats {
            weights[n_cats] = 1.0;
            n_cats += 1;
        } else {
            weights.truncate(n_cats);
            weights[k] += 1.0;
        }
    }
    // convert weights to counts, correcting for possible floating point
    // errors
    let counts: Vec<usize> =
        weights.iter().map(|w| (w + 0.5) as usize).collect();

    CrpDraw {
        asgn,
        counts,
        n_cats,
    }
}

/// The stick breaking algorithm has exceeded the max number of iterations.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheStickIsDust(u16);

/// Append new dirchlet weights by stick breaking until the new weight is less
/// than u*
///
/// **NOTE** This function is only for the slice reassignment kernel. It cuts out all
/// weights that are less that u*, so the sum of the weights will not be 1.
pub fn sb_slice_extend<R: Rng>(
    mut weights: Vec<f64>,
    alpha: f64,
    u_star: f64,
    mut rng: &mut R,
) -> Result<Vec<f64>, TheStickIsDust> {
    let mut b_star = weights.pop().unwrap();

    // If α is low and we do the dirichlet update w ~ Dir(n_1, ..., n_k, α),
    // the final weight will often be zero. In that case, we're done.
    if b_star <= 1E-16 {
        weights.push(b_star);
        return Ok(weights);
    }

    let beta = Beta::new(1.0, alpha).unwrap();

    let mut iters: u16 = 0;
    loop {
        let vk: f64 = beta.draw(&mut rng);
        let bk = vk * b_star;
        b_star *= 1.0 - vk;

        if bk >= u_star {
            weights.push(bk);
        }

        if b_star < u_star {
            return Ok(weights);
        }

        iters += 1;
        if iters > MAX_STICK_BREAKING_ITERS {
            return Err(TheStickIsDust(MAX_STICK_BREAKING_ITERS));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    mod sb_slice {
        use super::*;

        #[test]
        fn should_return_input_weights_if_alpha_is_zero() {
            let mut rng = rand::thread_rng();
            let weights_in: Vec<f64> = vec![0.8, 0.2, 0.0];
            let weights_out =
                sb_slice_extend(weights_in.clone(), 1.0, 0.2, &mut rng)
                    .unwrap();
            let good = weights_in
                .iter()
                .zip(weights_out.iter())
                .all(|(wi, wo)| (wi - wo).abs() < TOL);
            assert!(good);
        }

        #[test]
        fn should_return_error_for_zero_u_star() {
            let mut rng = rand::thread_rng();
            let weights_in: Vec<f64> = vec![0.8, 0.2];
            let u_star = 0.0;
            let res =
                sb_slice_extend(weights_in.clone(), 1.0, u_star, &mut rng);
            assert!(res.is_err());
        }

        #[test]
        fn smoke() {
            let mut rng = rand::thread_rng();
            let weights_in: Vec<f64> = vec![0.8, 0.2];
            let u_star = 0.1;
            let res =
                sb_slice_extend(weights_in.clone(), 1.0, u_star, &mut rng);
            assert!(res.is_ok());
        }
    }
}

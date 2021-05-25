//! Misc, generally useful helper functions
use braid_utils::Shape;
use rand::Rng;
use rv::misc::pflip;
use std::iter::Iterator;
use std::ops::Index;

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
    pub ncats: usize,
}

/// Draw from Chinese Restaraunt Process
pub fn crp_draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> CrpDraw {
    let mut ncats = 0;
    let mut weights: Vec<f64> = vec![];
    let mut asgn: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..n {
        weights.push(alpha);
        let k = pflip(&weights, 1, rng)[0];
        asgn.push(k);

        if k == ncats {
            weights[ncats] = 1.0;
            ncats += 1;
        } else {
            weights.truncate(ncats);
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
        ncats,
    }
}

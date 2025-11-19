//! Misc, generally useful helper functions
use std::iter::Iterator;

use rand::Rng;
use rv::misc::pflip;

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
        let k = pflip(&weights, Some(1.0), rng);
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

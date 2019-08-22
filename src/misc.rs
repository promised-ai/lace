//! Misc, generally useful helper functions
use std::iter::Iterator;

use rand::Rng;
use rv::misc::pflip;

// XXX: I kind of had rendering code as a string during build and injecting it
// here. It's probably better to just pick a strategy. If the build happens on
// a machine with different hardware, the benchmarks and the switching
// strategy won't make sense.
include!(concat!(env!("OUT_DIR"), "/msf_par_switch.rs"));

/// Draw n categorical indices in {0,..,k-1} from an n-by-k vector of vectors
/// of un-normalized log probabilities.
///
/// Automatically chooses whether to use serial or parallel computing.
pub fn massflip(logps: Vec<Vec<f64>>, mut rng: &mut impl Rng) -> Vec<usize> {
    if mfs_use_par(logps.len(), logps[0].len()) {
        braid_flippers::massflip_par(logps, &mut rng)
    } else {
        braid_flippers::massflip_ser(logps, &mut rng)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct CrpDraw {
    pub asgn: Vec<usize>,
    pub counts: Vec<usize>,
    pub ncats: usize,
}

/// Draw from Chinese Restaraunt Process
pub fn crp_draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> CrpDraw {
    let mut ncats = 1;
    let mut weights: Vec<f64> = vec![1.0];
    let mut asgn: Vec<usize> = Vec::with_capacity(n);

    asgn.push(0);

    for _ in 1..n {
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

/// A partition generator meant for testing. Partition of `n` points into
/// 1, ..., n partitions.
///
/// # Example
///
/// ```
/// # use braid::misc::Partition;
/// let partitions: Vec<Vec<usize>> = Partition::new(4).collect();
///
/// // Bell(4) = 15. There are 15 ways to partition 4 items.
/// assert_eq!(partitions.len(), 15);
///
/// assert_eq!(partitions[0], vec![0, 0, 0, 0]);
/// assert_eq!(partitions[14], vec![0, 1, 2, 3]);
/// ```
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Partition {
    z: Vec<usize>,
    k: Vec<usize>,
    n: usize,
    fresh: bool,
}

impl Partition {
    /// Create a generator that partitions `n` items
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::misc::Partition;
    /// let partitions: Vec<Vec<usize>> = Partition::new(4).collect();
    ///
    /// // Bell(4) = 15. There are 15 ways to partition 4 items.
    /// assert_eq!(partitions.len(), 15);
    /// ```
    pub fn new(n: usize) -> Self {
        Partition {
            z: vec![0; n],
            k: vec![0; n],
            n,
            fresh: true,
        }
    }
}

impl Iterator for Partition {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.fresh {
            self.fresh = false;
            Some(self.z.clone())
        } else {
            for i in (1..self.n).rev() {
                if self.z[i] <= self.k[i - 1] {
                    self.z[i] += 1;

                    if self.k[i] <= self.z[i] {
                        self.k[i] = self.z[i];
                    }

                    for j in (i + 1)..self.n {
                        self.z[j] = self.z[0];
                        self.k[j] = self.k[i];
                    }
                    return Some(self.z.clone());
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_iterator_creates_right_number_of_partitions() {
        // https://en.wikipedia.org/wiki/Bell_number
        let bell_nums: Vec<(usize, u64)> =
            vec![(1, 1), (2, 2), (3, 5), (4, 15), (5, 52), (6, 203)];

        for (n, bell) in bell_nums {
            let mut count: u64 = 0;
            Partition::new(n).for_each(|_| count += 1);
            assert_eq!(count, bell);
        }
    }
}

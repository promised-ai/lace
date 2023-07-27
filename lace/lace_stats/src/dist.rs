use crate::rv::traits::{HasSuffStat, Rv, SuffStat};

struct CanoncialDiscreteSuffStat {
    n: usize,
    counts: Vec<usize>,
}

impl SuffStat<usize> for CanoncialDiscreteSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn forget(&mut self, x: &usize) {
        if self.counts[*x] == 1 {
            // remove this
            self.counts.remove(*x);
        } else {
            // could should be greater than 1
            self.counts[*x] -= 1;
        }
    }

    fn observe(&mut self, x: &usize) {
        if *x >= self.k {
            unimplemented!()
        }
    }
}

struct CanonicalDiscrete {
    k: usize,
    alpha: f64,
    ln_weights: Vec<f64>,
}

impl Rv<usize> for CanonicalDiscrete {
    fn ln_f(&self, x: &usize) -> f64 {
        if x >=
    }
}

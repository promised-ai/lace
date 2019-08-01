use std::iter::Iterator;

mod sobol;
pub use sobol::*;

/// Halton pseudo-random sequence
pub struct HaltonSeq {
    dn: u32,
    nn: u32,
    base: u32,
}

impl HaltonSeq {
    pub fn new(base: u32) -> Self {
        HaltonSeq { nn: 0, dn: 1, base }
    }
}

impl Iterator for HaltonSeq {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.dn - self.nn;
        if x == 1 {
            self.nn = 1;
            self.dn *= self.base;
            return Some(self.nn as f64 / self.dn as f64);
        }

        let mut y = self.dn / self.base;
        while x < y {
            y /= self.base;
        }
        self.nn = (self.base + 1) * y - x;
        Some(self.nn as f64 / self.dn as f64)
    }
}

// FIXME: test sequence values

/// Convert a N-length vector xs = {x<sub>1</sub>, ..., x<sub>n</sub> :
/// x<sub>i</sub> ~ U(0, 1)} to a point on the N-simplex
///
/// # Example
///
/// Generate 100 quasi-random points on the 3-simplex
///
/// ```
/// # use braid_stats::seq::{SobolSeq, uvec_to_simplex};
/// SobolSeq::new(3)
///     .take(100)
///     .map(|uvec| uvec_to_simplex(uvec))
///     .for_each(|pt| {
///         assert_eq!(pt.len(), 3);
///         assert!( (pt.iter().sum::<f64>() - 1.0).abs() < 1e-6 );
///     })
/// ```
pub fn uvec_to_simplex(mut uvec: Vec<f64>) -> Vec<f64> {
    let n = uvec.len();
    uvec[n - 1] = 1.0;
    uvec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut um = uvec[0];

    for i in 1..n {
        let diff = uvec[i] - um;
        um = uvec[i];
        uvec[i] = diff;
    }
    uvec
}

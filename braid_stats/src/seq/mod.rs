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
            return Some(f64::from(self.nn) / f64::from(self.dn));
        }

        let mut y = self.dn / self.base;
        while x < y {
            y /= self.base;
        }
        self.nn = (self.base + 1) * y - x;
        Some(f64::from(self.nn) / f64::from(self.dn))
    }
}

// FIXME: test sequence values

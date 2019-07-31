/// Determine the location of the most significant on bit
fn most_significant_bit(n: u64) -> u32 {
    64 - n.leading_zeros()
}

fn find_first_zero(n: u64) -> u32 {
    (!n).trailing_zeros() + 1
}

const DIM_MAX: usize = 40;
const LOG_MAX: usize = 30;

const INITIAL_V: [[usize; DIM_MAX]; 8] = [
    [1; DIM_MAX],
    [
        0, 0, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3,
        3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3,
    ],
    [
        0, 0, 0, 7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1,
        7, 5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3,
    ],
    [
        0, 0, 0, 0, 0, 1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3,
        15, 7, 9, 13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17,
        1, 25, 29, 3, 31, 11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 33, 7, 5, 11, 39, 63, 27,
        17, 15, 23, 29, 3, 21, 13, 31, 25, 9, 49, 33, 19, 29, 11, 19, 27, 15,
        25,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 33, 115,
        41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3, 113, 61, 89, 45,
        107,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 23, 39,
    ],
];

const POLY: [u64; DIM_MAX] = [
    1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109, 103, 115,
    131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191, 253, 203,
    211, 239, 247, 285, 369, 299,
];

pub struct SobolSeq {
    dim: usize,
    lastq: [usize; DIM_MAX],
    recipd: f64,
    prev_seed: usize,
    v: [[usize; DIM_MAX]; LOG_MAX],
    started: bool,
}

impl SobolSeq {
    pub fn new(dim: usize) -> SobolSeq {
        let mut v: [[usize; DIM_MAX]; LOG_MAX] = [[0; DIM_MAX]; LOG_MAX];
        for i in 0..8 {
            v[i].clone_from_slice(&INITIAL_V[i]);
        }

        for vr in v.iter_mut() {
            vr[0] = 1;
        }

        // Initialize
        assert!(dim >= 1 && dim < DIM_MAX, "This Sobol sequence algorithm does not support the given dimension");

        for i in 2..=dim {
            // Determine the degree of the Polynomial given by POLY[i]
            let poly_degree = most_significant_bit(POLY[i - 1] >> 1) as usize;

            // Expand this pattern to separate components of the logical array INCLUD
            let mut j = POLY[i - 1];
            let includ: Vec<bool> = (1..=poly_degree)
                .rev()
                .map(|_| {
                    let j2 = j >> 1;
                    let out = j != 2 * j2;
                    j = j2;
                    out
                })
                .collect();

            // Calculate remaining elements of row i as said in Bratley and Fox, section 2.
            for j in (poly_degree + 1)..=LOG_MAX {
                let mut newv = v[j - poly_degree - 1][i - 1];
                let mut l = 1;
                for k in 1..=poly_degree {
                    l <<= 1;
                    if includ[k - 1] {
                        newv ^= l * v[j - k - 1][i - 1];
                        v[j - 1][i - 1] = newv;
                    }
                }
            }
        }

        // Multiply columns of V by appropiate power of two.
        let mut l = 1;
        for j in (1..LOG_MAX).rev() {
            l <<= 1;
            for d in 0..dim {
                v[j - 1][d] *= l;
            }
        }

        // RECIPD is 1/(common denominator of the elements in V)
        let recipd = 1.0 / (2 * l) as f64;
        let lastq = [0; DIM_MAX];

        SobolSeq {
            dim,
            prev_seed: 0,
            v,
            lastq,
            recipd,
            started: false,
        }
    }

    /// Update the seed used for generation
    pub fn set_seed(&mut self, seed: usize) {
        if seed == 0 {
            self.lastq = [0; DIM_MAX];
            self.prev_seed = 0;
        } else if seed <= self.prev_seed || self.prev_seed + 1 < seed {
            self.prev_seed = 0;
            self.lastq = [0; DIM_MAX];

            for seed_temp in 0..seed {
                let l = find_first_zero(seed_temp as u64) as usize;
                for i in 1..=self.dim {
                    self.lastq[i - 1] ^= self.v[l - 1][i - 1];
                }
            }
        }

        // set started to false to so this seed is used next time.
        self.started = false;
    }
}

impl Iterator for SobolSeq {
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        // If started, just increment the seed, otherwise, just use the seed given.
        let seed = if self.started {
            self.prev_seed + 1
        } else {
            self.started = true;
            self.prev_seed
        };
        let l: usize = find_first_zero(seed as u64) as usize;
        assert!(LOG_MAX >= l, "Too many calls to draw");

        let mut quasi: Vec<f64> = Vec::with_capacity(self.dim);

        // Generate the next element's part, then update lastq
        for d in 1..=self.dim {
            quasi.push((self.lastq[d - 1] as f64) * self.recipd);
            self.lastq[d - 1] ^= self.v[l - 1][d - 1];
        }

        // Increment the seed
        self.prev_seed = seed;
        Some(quasi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{abs_diff_eq, assert_ulps_eq};
    use rv::misc::ks_test;

    #[test]
    fn most_significant_bit_checks() {
        assert_eq!(most_significant_bit(0), 0);
        assert_eq!(most_significant_bit(1), 1);
        assert_eq!(most_significant_bit(2), 2);
        assert_eq!(most_significant_bit(3), 2);
        assert_eq!(most_significant_bit(4), 3);
        assert_eq!(most_significant_bit(5), 3);
        assert_eq!(most_significant_bit(6), 3);
        assert_eq!(most_significant_bit(7), 3);
        assert_eq!(most_significant_bit(8), 4);
        assert_eq!(most_significant_bit(9), 4);
        assert_eq!(most_significant_bit(10), 4);
        assert_eq!(most_significant_bit(11), 4);
        assert_eq!(most_significant_bit(12), 4);
        assert_eq!(most_significant_bit(13), 4);
        assert_eq!(most_significant_bit(14), 4);
        assert_eq!(most_significant_bit(15), 4);
        assert_eq!(most_significant_bit(16), 5);
        assert_eq!(most_significant_bit(17), 5);
        assert_eq!(most_significant_bit(1023), 10);
        assert_eq!(most_significant_bit(1024), 11);
        assert_eq!(most_significant_bit(1024), 11);
        assert_eq!(most_significant_bit(0xffffffffffffffff), 64);
    }

    #[test]
    fn find_first_zero_checks() {
        assert_eq!(find_first_zero(0), 1);
        assert_eq!(find_first_zero(1), 2);
        assert_eq!(find_first_zero(2), 1);
        assert_eq!(find_first_zero(3), 3);
        assert_eq!(find_first_zero(4), 1);
        assert_eq!(find_first_zero(5), 2);
        assert_eq!(find_first_zero(6), 1);
        assert_eq!(find_first_zero(7), 4);
        assert_eq!(find_first_zero(8), 1);
        assert_eq!(find_first_zero(9), 2);
        assert_eq!(find_first_zero(10), 1);
        assert_eq!(find_first_zero(11), 3);
        assert_eq!(find_first_zero(12), 1);
        assert_eq!(find_first_zero(13), 2);
        assert_eq!(find_first_zero(14), 1);
        assert_eq!(find_first_zero(15), 5);
        assert_eq!(find_first_zero(16), 1);
        assert_eq!(find_first_zero(17), 2);
        assert_eq!(find_first_zero(1023), 11);
        assert_eq!(find_first_zero(1024), 1);
        assert_eq!(find_first_zero(1025), 2);
    }

    #[test]
    fn sobol_sequence_uniform() {
        let s = SobolSeq::new(1);
        let seq: Vec<f64> = s.take(1000).map(|v| *v.get(0).unwrap()).collect();
        let (_, pvalue) = ks_test(&seq, |x| x);
        abs_diff_eq!(pvalue, 1.0);
    }

    #[test]
    fn sobol_sequence_matches_reference() {
        let s = SobolSeq::new(2);
        let r: Vec<Vec<f64>> = s.take(10).collect();

        let expected = vec![
            vec![0., 0.],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.25, 0.75],
            vec![0.375, 0.375],
            vec![0.875, 0.875],
            vec![0.625, 0.125],
            vec![0.125, 0.625],
            vec![0.1875, 0.3125],
            vec![0.6875, 0.8125],
        ];
        assert_eq!(r, expected);
    }

    #[test]
    fn sobol_sequence_set_seed() {
        let mut s = SobolSeq::new(1);
        // Update the seed by taking elements from the sequence
        s.next();
        s.next();
        s.set_seed(0);
        assert_eq!(s.next(), Some(vec![0.0]));

        s.set_seed(5);
        assert_eq!(s.next(), Some(vec![0.875]));
    }

    /// Test that this sequence returns an approximate value of pi
    #[test]
    fn pi_approx() {
        let s = SobolSeq::new(2);
        let size: usize = 1_000_000;
        let inside = s.take(size).fold(0_usize, |sum, r| {
            let x = r.get(0).unwrap();
            let y = r.get(1).unwrap();
            if x * x + y * y < 1.0 {
                sum + 1
            } else {
                sum
            }
        });

        let pi_approx = 4.0 * (inside as f64) / (size as f64);
        assert_ulps_eq!(pi_approx, std::f64::consts::PI, epsilon = 1e-3);
    }

    /// Test approximating $\int_0^1 \int_0^1 \sin(\pi * x) \sin(\pi y) dx dy$
    #[test]
    fn two_sin_integral() {
        use std::f64::consts::PI;
        let s = SobolSeq::new(2);
        let size: usize = 1_000_000;

        let sum = s.take(size).fold(0_f64, |sum, r| {
            let x = r.get(0).unwrap();
            let y = r.get(1).unwrap();
            sum + (PI * x).sin() * (PI * y).sin()
        });

        let est = sum / (size as f64);
        assert_ulps_eq!(est, 4.0 / (PI * PI), epsilon = 1e-4);
    }
}

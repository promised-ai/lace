use std::f64::NEG_INFINITY;

use braid_utils::Matrix;
use rand::Rng;
use rayon::prelude::*;

pub fn massflip_mat<R: Rng>(logps: &Matrix<f64>, rng: &mut R) -> Vec<usize> {
    if logps.ncols() == 1 {
        panic!("K should never be 1")
    }

    let nrows = logps.nrows();
    let ncols = logps.ncols();

    (0..nrows)
        .map(|i| {
            let logp0 = logps[(i, 0)];
            let mut ps: Vec<f64> = Vec::with_capacity(ncols);
            ps.push(logp0);

            let maxval = (1..ncols).fold(logp0, |max, j| {
                let logp = logps[(i, j)];
                ps.push(logp);
                if logp > max {
                    logp
                } else {
                    max
                }
            });

            ps[0] = (logp0 - maxval).exp();
            (1..ncols).for_each(|j| {
                let p = (ps[j] - maxval).exp() + ps[j - 1];
                ps[j] = p;
            });

            let r: f64 = rng.gen::<f64>() * ps[ncols - 1];

            ps.iter().fold(0_u16, |acc, p| acc + (*p < r) as u16) as usize
        })
        .collect()
}

pub fn massflip_mat_par<R: Rng>(
    logps: &Matrix<f64>,
    rng: &mut R,
) -> Vec<usize> {
    if logps.ncols() == 1 {
        panic!("K should never be 1")
    }

    let ncols = logps.ncols();

    let rs: Vec<f64> = (0..logps.nrows()).map(|_| rng.gen::<f64>()).collect();

    rs.par_iter()
        .enumerate()
        .map(|(i, &u)| {
            let logp0 = logps[(i, 0)];
            let mut ps: Vec<f64> = Vec::with_capacity(ncols);
            ps.push(logp0);

            let maxval = (1..ncols).fold(logp0, |max, j| {
                let logp = logps[(i, j)];
                ps.push(logp);
                if logp > max {
                    logp
                } else {
                    max
                }
            });

            // There should always be at least two columns
            ps[0] = (logp0 - maxval).exp();
            ps[1] = (ps[1] - maxval).exp() + ps[0];
            (2..ncols).for_each(|j| {
                let p = (ps[j] - maxval).exp() + ps[j - 1];
                ps[j] = p;
            });

            let r: f64 = u * ps[ncols - 1];

            ps.iter().fold(0_u16, |acc, p| acc + (*p < r) as u16) as usize
        })
        .collect()
}

pub fn massflip_slice_mat<R: Rng>(
    logps: &Matrix<f64>,
    rng: &mut R,
) -> Vec<usize> {
    let nrows = logps.nrows();
    let ncols = logps.ncols();

    (0..nrows)
        .map(|i| {
            let maxval = (1..ncols).fold(logps[(i, 0)], |max, j| {
                let val = logps[(i, j)];
                if val > max {
                    val
                } else {
                    max
                }
            });

            // XXX: using is lps[i] != NEG_INFINITY saves 2 EQ comparisons, two ORs,
            // and one NOT compared to lps[i].is_finite(). We only care whether the
            // entry is log(0) == NEG_INFINITY. If something is NAN of Inf, then we
            // have other problems.
            let mut ps: Vec<f64> = Vec::with_capacity(ncols);
            (0..ncols).fold(0.0, |prev, j| {
                let logp = logps[(i, j)];
                let value = if logp != NEG_INFINITY {
                    (logp - maxval).exp() + prev
                } else {
                    prev
                };
                ps.push(value);
                value
            });

            let scale: f64 = ps[ncols - 1];
            let r: f64 = rng.gen::<f64>() * scale;

            ps.iter()
                .fold(0_usize, |acc, p| if *p < r { acc + 1 } else { acc })
        })
        .collect()
}

pub fn massflip_slice_mat_par<R: Rng>(
    logps: &Matrix<f64>,
    rng: &mut R,
) -> Vec<usize> {
    let nrows = logps.nrows();
    let ncols = logps.ncols();

    let us: Vec<f64> = (0..nrows).map(|_| rng.gen::<f64>()).collect();

    us.par_iter()
        .enumerate()
        .map(|(i, &u)| {
            let maxval = (1..ncols).fold(logps[(i, 0)], |max, j| {
                let val = logps[(i, j)];
                if val > max {
                    val
                } else {
                    max
                }
            });

            // XXX: using is lps[i] != NEG_INFINITY saves 2 EQ comparisons, two ORs,
            // and one NOT compared to lps[i].is_finite(). We only care whether the
            // entry is log(0) == NEG_INFINITY. If something is NAN of Inf, then we
            // have other problems.
            let mut ps: Vec<f64> = Vec::with_capacity(ncols);
            (0..ncols).fold(0.0, |prev, j| {
                let logp = logps[(i, j)];
                let value = if logp != NEG_INFINITY {
                    (logp - maxval).exp() + prev
                } else {
                    prev
                };
                ps.push(value);
                value
            });

            let scale: f64 = ps[ncols - 1];
            let r: f64 = u * scale;

            ps.iter()
                .fold(0_usize, |acc, p| if *p < r { acc + 1 } else { acc })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    fn gen_weights(nrows: usize, ncols: usize) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256Plus::seed_from_u64(1337);
        let logps: Vec<Vec<f64>> = (0..nrows)
            .map(|_| {
                let mut ps: Vec<f64> =
                    (0..ncols).map(|_| rng.gen::<f64>()).collect();
                let sum: f64 = ps.iter().sum::<f64>();
                ps.drain(..).map(|p| (p / sum).ln()).collect::<Vec<f64>>()
            })
            .collect();

        logps
    }

    // stupid, slow, simple version
    pub fn massflip_naive<R: Rng>(
        logps: &Vec<Vec<f64>>,
        rng: &mut R,
    ) -> Vec<usize> {
        logps
            .iter()
            .map(|row| {
                let mut ps: Vec<f64> = row
                    .iter()
                    .scan(0.0, |state, &logp| {
                        *state += logp.exp();
                        Some(*state)
                    })
                    .collect();

                let z = ps[ps.len() - 1];

                ps.iter_mut().for_each(|p| *p /= z);

                let u: f64 = rng.gen();
                ps.iter().enumerate().find(|(_, &p)| p > u).unwrap().0
            })
            .collect()
    }

    #[test]
    fn naive_matches_matrix_for_same_seed() {
        let logps = gen_weights(100, 5);
        let ixs_mat = {
            // will be transposed inside
            let logps_m = Matrix::from_vecs(&logps);
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            massflip_mat_par(&logps_m, &mut rng)
        };

        let ixs_naive = {
            // will be transposed inside
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            massflip_naive(&logps, &mut rng)
        };

        assert_eq!(ixs_mat.len(), ixs_naive.len());

        for (a, b) in ixs_mat.iter().zip(ixs_naive.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn serial_and_par_matrix_are_the_same_for_same_seed() {
        let logps = gen_weights(100, 5);
        let ixs_ser = {
            // will be transposed inside
            let logps_m = Matrix::from_vecs(&logps);
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            massflip_mat_par(&logps_m, &mut rng)
        };

        let ixs_par = {
            // will be transposed inside
            let logps_m = Matrix::from_vecs(&logps);
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            massflip_mat(&logps_m, &mut rng)
        };

        assert_eq!(ixs_par.len(), ixs_ser.len());

        for (a, b) in ixs_ser.iter().zip(ixs_par.iter()) {
            assert_eq!(a, b);
        }
    }
}

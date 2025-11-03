use std::f64::NEG_INFINITY;
use std::ops::Index;

use lace_stats::rand::Rng;
use lace_utils::Shape;
use rayon::prelude::*;

/// Draw n categorical indices in {0,..,k-1} from an n-by-k vector of vectors
/// of un-normalized log probabilities.
///
/// Automatically chooses whether to use serial or parallel computing.
pub fn massflip<M>(logps: M, mut rng: &mut impl Rng) -> Vec<usize>
where
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    massflip_mat_par(logps, &mut rng)
}

pub fn massflip_mat<M, R>(logps: M, rng: &mut R) -> Vec<usize>
where
    R: Rng,
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    if logps.n_cols() == 1 {
        panic!("K should never be 1")
    }

    let n_rows = logps.n_rows();
    let n_cols = logps.n_cols();

    (0..n_rows)
        .map(|i| {
            let logp0 = logps[(i, 0)];
            let mut ps: Vec<f64> = Vec::with_capacity(n_cols);
            ps.push(logp0);

            let maxval = (1..n_cols).fold(logp0, |max, j| {
                let logp = logps[(i, j)];
                ps.push(logp);
                if logp > max {
                    logp
                } else {
                    max
                }
            });

            ps[0] = (logp0 - maxval).exp();
            (1..n_cols).for_each(|j| {
                let p = (ps[j] - maxval).exp() + ps[j - 1];
                ps[j] = p;
            });

            let r: f64 = rng.random::<f64>() * ps[n_cols - 1];

            ps.iter().fold(0_u16, |acc, p| acc + (*p < r) as u16) as usize
        })
        .collect()
}

pub fn massflip_mat_par<M, R>(logps: M, rng: &mut R) -> Vec<usize>
where
    R: Rng,
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    if logps.n_cols() == 1 {
        panic!("K should never be 1")
    }

    let n_cols = logps.n_cols();

    let rs: Vec<f64> =
        (0..logps.n_rows()).map(|_| rng.random::<f64>()).collect();

    rs.par_iter()
        .enumerate()
        .map(|(i, &u)| {
            let logp0 = logps[(i, 0)];
            let mut ps: Vec<f64> = Vec::with_capacity(n_cols);
            ps.push(logp0);

            let maxval = (1..n_cols).fold(logp0, |max, j| {
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
            (2..n_cols).for_each(|j| {
                let p = (ps[j] - maxval).exp() + ps[j - 1];
                ps[j] = p;
            });

            let r: f64 = u * ps[n_cols - 1];

            ps.iter().fold(0_u16, |acc, p| acc + (*p < r) as u16) as usize
        })
        .collect()
}

pub fn massflip_slice_mat<M, R>(logps: M, rng: &mut R) -> Vec<usize>
where
    R: Rng,
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    let n_rows = logps.n_rows();
    let n_cols = logps.n_cols();

    (0..n_rows)
        .map(|i| {
            let maxval = (1..n_cols).fold(logps[(i, 0)], |max, j| {
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
            let mut ps: Vec<f64> = Vec::with_capacity(n_cols);
            (0..n_cols).fold(0.0, |prev, j| {
                let logp = logps[(i, j)];
                let value = if logp != NEG_INFINITY {
                    (logp - maxval).exp() + prev
                } else {
                    prev
                };
                ps.push(value);
                value
            });

            let scale: f64 = ps[n_cols - 1];
            let r: f64 = rng.random::<f64>() * scale;

            ps.iter()
                .fold(0_usize, |acc, p| if *p < r { acc + 1 } else { acc })
        })
        .collect()
}

pub fn massflip_slice_mat_par<M, R>(logps: M, rng: &mut R) -> Vec<usize>
where
    R: Rng,
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    let n_rows = logps.n_rows();
    let n_cols = logps.n_cols();

    let us: Vec<f64> = (0..n_rows).map(|_| rng.random::<f64>()).collect();

    us.par_iter()
        .enumerate()
        .map(|(i, &u)| {
            let maxval = (1..n_cols).fold(logps[(i, 0)], |max, j| {
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
            let mut ps: Vec<f64> = Vec::with_capacity(n_cols);
            (0..n_cols).fold(0.0, |prev, j| {
                let logp = logps[(i, j)];
                let value = if logp != NEG_INFINITY {
                    (logp - maxval).exp() + prev
                } else {
                    prev
                };
                ps.push(value);
                value
            });

            let scale: f64 = ps[n_cols - 1];
            let r: f64 = u * scale;

            ps.iter()
                .fold(0_usize, |acc, p| if *p < r { acc + 1 } else { acc })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lace_stats::rand::SeedableRng;
    use lace_utils::Matrix;
    use rand_xoshiro::Xoshiro256Plus;

    fn gen_weights(n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256Plus::seed_from_u64(1337);
        let logps: Vec<Vec<f64>> = (0..n_rows)
            .map(|_| {
                let mut ps: Vec<f64> =
                    (0..n_cols).map(|_| rng.random::<f64>()).collect();
                let sum: f64 = ps.iter().sum::<f64>();
                ps.drain(..).map(|p| (p / sum).ln()).collect::<Vec<f64>>()
            })
            .collect();

        logps
    }

    // stupid, slow, simple version
    pub fn massflip_naive<'a, A, B, R>(logps: A, rng: &mut R) -> Vec<usize>
    where
        A: IntoIterator<Item = B>,
        B: IntoIterator<Item = &'a f64>,
        R: Rng,
    {
        logps
            .into_iter()
            .map(|row| {
                let mut ps: Vec<f64> = row
                    .into_iter()
                    .scan(0.0, |state, &logp| {
                        *state += logp.exp();
                        Some(*state)
                    })
                    .collect();

                let z = ps[ps.len() - 1];

                ps.iter_mut().for_each(|p| *p /= z);

                let u: f64 = rng.random();
                ps.iter().enumerate().find(|(_, &p)| p > u).unwrap().0
            })
            .collect()
    }

    #[test]
    fn naive_matches_matrix_for_same_seed() {
        let logps = gen_weights(100, 5);
        let ixs_mat = {
            // will be transposed inside
            let logps_m = Matrix::from_vecs(logps.clone());
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
            let logps_m = Matrix::from_vecs(logps.clone()).implicit_transpose();
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            massflip_mat_par(&logps_m, &mut rng)
        };

        let ixs_par = {
            // will be transposed inside
            let logps_m = Matrix::from_vecs(logps).implicit_transpose();
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            massflip_mat(&logps_m, &mut rng)
        };

        assert_eq!(ixs_par.len(), ixs_ser.len());

        for (a, b) in ixs_ser.iter().zip(ixs_par.iter()) {
            assert_eq!(a, b);
        }
    }
}

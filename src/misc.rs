extern crate rand;
extern crate rv;

use std::collections::{BTreeMap, HashSet};
use std::f64::{NAN, NEG_INFINITY};
use std::iter::FromIterator;
use std::iter::Iterator;
use std::mem::swap;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use self::rv::misc::pflip;
use rayon::prelude::*;
use std::cmp::PartialOrd;
use std::ops::AddAssign;
use std::str::FromStr;

/// Attempt to turn a `&str` into a `T`
pub fn parse_result<T: FromStr>(res: &str) -> Option<T> {
    // For csv, empty cells are considered missing regardless of type
    if res.is_empty() {
        None
    } else {
        match res.parse::<T>() {
            Ok(x) => Some(x),
            Err(_) => panic!("Could not parse \"{}\"", res),
        }
    }
}

/// Like `signum`, but return 0.0 if the number is zero
pub fn sign(x: f64) -> f64 {
    if x.is_nan() {
        NAN
    } else if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// The mean of a vector of f64
pub fn mean(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    xs.iter().fold(0.0, |acc, x| x + acc) / n
}

/// The variance of a vector of f64
pub fn var(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    let m = mean(xs);
    let v = xs.iter().fold(0.0, |acc, x| acc + (x - m) * (x - m));
    // TODO: Add dof and return 0 if n == 1
    v / n
}

/// The standard deviation of a vector of f64
pub fn std(xs: &[f64]) -> f64 {
    let v: f64 = var(xs);
    v.sqrt()
}

/// Bins the entries in `xs` into `k` bins.
///
/// # Example
///
/// ```rust
/// # extern crate braid;
/// # use braid::misc::bincount;
/// let xs: Vec<usize> = vec![0, 0, 1, 2, 2, 2, 3];
///
/// assert_eq!(bincount(&xs, 4), vec![2, 1, 3, 1]);
/// ```
pub fn bincount<T>(xs: &[T], k: usize) -> Vec<usize>
where
    T: Clone + Into<usize>,
{
    let mut counts = vec![0; k];
    xs.iter().for_each(|x| {
        // TODO: I hate this clone
        let ix: usize = (*x).clone().into();
        counts[ix] += 1;
    });
    counts
}

/// Cumulative sum of `xs`
pub fn cumsum<T>(xs: &[T]) -> Vec<T>
where
    T: AddAssign + Clone,
{
    let mut summed: Vec<T> = xs.to_vec();
    for i in 1..xs.len() {
        summed[i] += summed[i - 1].clone();
    }
    summed
}

/// Returns the index of the largest element in xs.
///
/// If there are multiple largest elements, returns the index of the first.
pub fn argmax<T: PartialOrd>(xs: &[T]) -> usize {
    if xs.is_empty() {
        panic!("Empty container");
    }

    if xs.len() == 1 {
        0
    } else {
        let mut maxval = &xs[0];
        let mut max_ix: usize = 0;
        for i in 1..xs.len() {
            let x = &xs[i];
            if x > maxval {
                maxval = x;
                max_ix = i;
            }
        }
        max_ix
    }
}

/// Returns the index of the smallest element in xs.
///
/// If there are multiple smallest elements, returns the index of the first.
pub fn argmin<T: PartialOrd>(xs: &[T]) -> usize {
    if xs.is_empty() {
        panic!("Empty container");
    }

    if xs.len() == 1 {
        0
    } else {
        let mut minval = &xs[0];
        let mut min_ix: usize = 0;
        for i in 1..xs.len() {
            let x = &xs[i];
            if x < minval {
                minval = x;
                min_ix = i;
            }
        }
        min_ix
    }
}

// XXX: This is not optimized. If we compare pairs of element, we get 1.5n
// comparisons instead of 2n.
/// Returns a tuple (min_elem, max_elem).
///
/// Faster than calling min and max individually
pub fn minmax<T: PartialOrd + Clone>(xs: &[T]) -> (T, T) {
    if xs.is_empty() {
        panic!("Empty slice");
    }

    if xs.len() == 1 {
        return (xs[0].clone(), xs[0].clone());
    }

    let mut min = &xs[0];
    let mut max = &xs[1];

    if min > max {
        swap(&mut min, &mut max);
    }

    for i in 2..xs.len() {
        if xs[i] > *max {
            max = &xs[i];
        } else if xs[i] < *min {
            min = &xs[i];
        }
    }

    (min.clone(), max.clone())
}

/// Numerically stable `log(sum(exp(xs))`
pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        panic!("Empty container");
    } else if xs.len() == 1 {
        xs[0]
    } else {
        let maxval =
            *xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        xs.iter().fold(0.0, |acc, x| acc + (x - maxval).exp()).ln() + maxval
    }
}

/// Choose two distinct random numbers in [0, ..., n-1]
pub fn choose2ixs<R: Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    if n < 2 {
        panic!("n must be 2 or greater")
    } else if n == 2 {
        (0, 1)
    } else {
        let i: usize = rng.gen_range(0, n);
        loop {
            let j: usize = rng.gen_range(0, n);
            if j != i {
                return (i, j);
            }
        }
    }
}

pub fn massflip_par<R: Rng>(
    mut logps: Vec<Vec<f64>>,
    rng: &mut R,
) -> Vec<usize> {
    let n = logps.len();
    let k = logps[0].len();
    let u = Uniform::new(0.0, 1.0);
    let us: Vec<f64> = (0..n).map(|_| rng.sample(u)).collect();

    let mut out: Vec<usize> = Vec::with_capacity(n);
    logps
        .par_iter_mut()
        .zip_eq(us.par_iter())
        .map(|(lps, u)| {
            let maxval = lps.iter().fold(NEG_INFINITY, |max, &val| {
                if val > max {
                    val
                } else {
                    max
                }
            });
            lps[0] -= maxval;
            lps[0] = lps[0].exp();
            for i in 1..k {
                lps[i] -= maxval;
                lps[i] = lps[i].exp();
                lps[i] += lps[i - 1]
            }

            let r = u * *lps.last().unwrap();

            // Is a for loop faster?
            lps.iter().fold(0, |acc, &p| acc + ((p < r) as usize))
        })
        .collect_into_vec(&mut out);
    out
}

pub fn massflip(mut logps: Vec<Vec<f64>>, rng: &mut impl Rng) -> Vec<usize> {
    let k = logps[0].len();
    let mut ixs: Vec<usize> = Vec::with_capacity(logps.len());
    let u = Uniform::new(0.0, 1.0);

    for lps in &mut logps {
        // ixs.push(log_pflip(&lps, &mut rng)); // debug
        let maxval =
            lps.iter().fold(
                NEG_INFINITY,
                |max, &val| {
                    if val > max {
                        val
                    } else {
                        max
                    }
                },
            );
        lps[0] -= maxval;
        lps[0] = lps[0].exp();
        for i in 1..k {
            lps[i] -= maxval;
            lps[i] = lps[i].exp();
            lps[i] += lps[i - 1]
        }

        let scale: f64 = *lps.last().unwrap();
        let r: f64 = rng.sample(u) * scale;

        let mut ct: usize = 0;
        for p in lps {
            ct += (*p < r) as usize;
        }
        ixs.push(ct);
    }
    ixs
}

pub fn massflip_flat(
    mut logps: Vec<f64>,
    n: usize,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let mut ixs: Vec<usize> = Vec::with_capacity(logps.len());
    let mut a = 0;
    let u = Uniform::new(0.0, 1.0);
    while a < n * k {
        let b = a + k - 1;
        let maxval: f64 = *logps[a..b]
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        logps[a] -= maxval;
        logps[a] = logps[a].exp();
        for j in a + 1..b {
            logps[j] -= maxval;
            logps[j] = logps[j].exp();
            logps[j] += logps[j - 1]
        }
        let scale: f64 = logps[b];
        let r: f64 = rng.sample(u) * scale;

        let mut ct: usize = 0;
        for p in logps[a..b].iter() {
            ct += (*p < r) as usize;
        }
        ixs.push(ct);
        a += k;
    }
    ixs
}

// FIXME: World's crappiest transpose
pub fn transpose(mat_in: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let nrows = mat_in.len();
    let ncols = mat_in[0].len();
    let mut mat_out: Vec<Vec<f64>> = vec![vec![0.0; nrows]; ncols];

    for (i, row) in mat_in.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            mat_out[j][i] = x;
        }
    }

    mat_out
}

/// Returns a vector, in descending order, of the indices of the unused
/// components in `asgn_vec`, which can take on values from 0...k-1
pub fn unused_components(k: usize, asgn_vec: &[usize]) -> Vec<usize> {
    let all_cpnts: HashSet<_> = HashSet::from_iter(0..k);
    let used_cpnts = HashSet::from_iter(asgn_vec.iter().cloned());
    let mut unused_cpnts: Vec<&usize> =
        all_cpnts.difference(&used_cpnts).collect();
    unused_cpnts.sort();
    // needs to be in reverse order, because we want to remove the
    // higher-indexed views first to minimize bookkeeping.
    unused_cpnts.reverse();
    unused_cpnts.iter().map(|&z| *z).collect()
}

/// The number of unique values in `xs`
pub fn n_unique(xs: &Vec<f64>, cutoff: usize) -> usize {
    let mut unique: Vec<f64> = vec![xs[0]];
    for x in xs.iter().skip(1) {
        if !unique.iter().any(|y| y == x) {
            unique.push(*x);
        }
        if unique.len() > cutoff {
            return unique.len();
        }
    }
    unique.len()
}

/// Turn `Vec<Map<K, V>>` into `Map<K, Vec<V>>`
pub fn transpose_mapvec<K: Clone + Ord, V: Clone>(
    mapvec: &Vec<BTreeMap<K, V>>,
) -> BTreeMap<K, Vec<V>> {
    let mut transposed: BTreeMap<K, Vec<V>> = BTreeMap::new();
    let n = mapvec.len();

    for key in mapvec[0].keys() {
        transposed.insert(key.clone(), Vec::with_capacity(n));
    }

    for row in mapvec {
        for (key, value) in row {
            transposed.get_mut(key).unwrap().push(value.clone());
        }
    }

    transposed
}

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

// A partition generator meant for testing
#[derive(Clone, Debug, Hash)]
pub struct Partition {
    z: Vec<usize>,
    k: Vec<usize>,
    n: usize,
    fresh: bool,
}

impl Partition {
    pub fn new(n: usize) -> Self {
        Partition {
            z: vec![0; n],
            k: vec![0; n],
            n: n,
            fresh: true,
        }
    }

    pub fn partition(&self) -> &Vec<usize> {
        &self.z
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

/// Factorial, n!
pub fn factorial(n: u64) -> u64 {
    (1..=n).fold(1, |acc, k| acc * k)
}

/// Binomial coefficient, n choose k
pub fn binom(n: u64, k: u64) -> u64 {
    if k < 1 {
        1
    } else if k == 1 || n - k == 1 {
        n
    } else if n - k > k {
        let numer = (n - k + 1..=n).fold(1, |acc, x| acc * x);
        numer / factorial(k)
    } else {
        let numer = (k + 1..=n).fold(1, |acc, x| acc * x);
        numer / factorial(n - k)
    }
}

/// Sterling number of the 2nd kind
///
/// The number of ways to partition n items into k subsets
pub fn sterling(n: u64, k: u64) -> u64 {
    let sum: u64 = (0..=k).fold(0_i64, |acc, j| {
        let a = (-1_i64).pow((k - j) as u32);
        let b = binom(k, j) as i64;
        let c = (j as i64).pow(n as u32);
        acc + a * b * c
    }) as u64;
    sum / factorial(k)
}

/// The number of ways to partition n items into 1...n subsets
pub fn bell(n: u64) -> u64 {
    (0..=n).fold(0_u64, |acc, k| acc + sterling(n, k))
}

/// The number of bi-partitions of an n-by-m (rows-by-columns) table
pub fn ccnum(n: u64, m: u64) -> u64 {
    (0..=m).fold(0_u64, |acc, k| acc + sterling(m, k) * bell(n).pow(k as u32))
}

#[cfg(test)]
mod tests {
    use self::rand::chacha::ChaChaRng;
    use self::rand::FromEntropy;
    use super::*;

    const TOL: f64 = 1E-10;

    // mean
    // ----
    #[test]
    fn mean_1() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(mean(&xs), 2.0, epsilon = 10E-10);
    }

    #[test]
    fn mean_2() {
        let xs: Vec<f64> = vec![1.0 / 3.0, 2.0 / 3.0, 5.0 / 8.0, 11.0 / 12.0];
        assert_relative_eq!(mean(&xs), 0.63541666666666663, epsilon = 10E-8);
    }

    #[test]
    fn var_1() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(var(&xs), 2.0, epsilon = 10E-10);
    }

    #[test]
    fn var_2() {
        let xs: Vec<f64> = vec![1.0 / 3.0, 2.0 / 3.0, 5.0 / 8.0, 11.0 / 12.0];
        assert_relative_eq!(var(&xs), 0.04286024305555555, epsilon = 10E-8);
    }

    // cumsum
    // ------
    #[test]
    fn cumsum_should_work_on_u8() {
        let xs: Vec<u8> = vec![2, 3, 4, 1, 0];
        assert_eq!(cumsum(&xs), [2, 5, 9, 10, 10]);
    }

    #[test]
    fn cumsum_should_work_on_u16() {
        let xs: Vec<u16> = vec![2, 3, 4, 1, 0];
        assert_eq!(cumsum(&xs), [2, 5, 9, 10, 10]);
    }

    #[test]
    fn cumsum_should_work_on_f64() {
        let xs: Vec<f64> = vec![2.0, 3.0, 4.0, 1.0, 0.1];
        assert_eq!(cumsum(&xs), [2.0, 5.0, 9.0, 10.0, 10.1]);
    }

    #[test]
    fn cumsum_should_work_do_nothing_to_one_length_vector() {
        let xs: Vec<u8> = vec![2];
        assert_eq!(cumsum(&xs), [2]);
    }

    #[test]
    fn cumsum_should_return_empty_if_given_empty() {
        let xs: Vec<f64> = Vec::new();
        assert!(cumsum(&xs).is_empty());
    }

    // argmax
    // ------
    #[test]
    fn argmax_should_work_on_unique_values() {
        let xs: Vec<f64> = vec![2.0, 3.0, 4.0, 1.0, 0.1];
        assert_eq!(argmax(&xs), 2);
    }

    #[test]
    fn argmax_should_return_0_if_max_value_is_in_0_index() {
        let xs: Vec<f64> = vec![20.0, 3.0, 4.0, 1.0, 0.1];
        assert_eq!(argmax(&xs), 0);
    }

    #[test]
    fn argmax_should_return_last_index_if_max_value_is_last() {
        let xs: Vec<f64> = vec![0.0, 3.0, 4.0, 1.0, 20.1];
        assert_eq!(argmax(&xs), 4);
    }

    #[test]
    fn argmax_should_return_index_of_first_max_value_if_repeats() {
        let xs: Vec<f64> = vec![0.0, 0.0, 2.0, 1.0, 2.0];
        assert_eq!(argmax(&xs), 2);
    }

    #[test]
    #[should_panic]
    fn argmax_should_panic_given_empty_container() {
        let xs: Vec<f64> = Vec::new();
        argmax(&xs);
    }

    // argmin
    // ------
    #[test]
    fn argmin_normal() {
        let xs: Vec<f64> = vec![2.0, 3.0, 4.0, 1.0, 0.1];
        assert_eq!(argmin(&xs), 4);
    }

    #[test]
    fn argmin_should_return_0_if_min_value_is_in_0_index() {
        let xs: Vec<f64> = vec![0.001, 3.0, 4.0, 1.0, 0.1];
        assert_eq!(argmin(&xs), 0);
    }

    #[test]
    fn argmin_should_return_last_index_if_min_value_is_last() {
        let xs: Vec<f64> = vec![1.0, 3.0, 4.0, 1.0, 0.001];
        assert_eq!(argmin(&xs), 4);
    }

    #[test]
    fn argmin_should_return_index_of_first_min_value_if_repeats() {
        let xs: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 2.0];
        assert_eq!(argmin(&xs), 1);
    }

    #[test]
    #[should_panic]
    fn argmin_should_panic_given_empty_container() {
        let xs: Vec<f64> = Vec::new();
        argmin(&xs);
    }

    // minmax
    // ------
    #[test]
    fn minmax_should_copy_the_entry_for_a_single_element_slice() {
        let xs: Vec<u8> = vec![1];
        let (a, b) = minmax(&xs);

        assert_eq!(a, 1);
        assert_eq!(b, 1);
    }

    #[test]
    fn minmax_should_sort_two_element_slice_1() {
        let xs: Vec<u8> = vec![1, 2];
        let (a, b) = minmax(&xs);

        assert_eq!(a, 1);
        assert_eq!(b, 2);
    }

    #[test]
    fn minmax_should_sort_two_element_slice_2() {
        let xs: Vec<u8> = vec![2, 1];
        let (a, b) = minmax(&xs);

        assert_eq!(a, 1);
        assert_eq!(b, 2);
    }

    #[test]
    fn minmax_on_sorted_unique_slice() {
        let xs: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let (a, b) = minmax(&xs);

        assert_eq!(a, 0);
        assert_eq!(b, 5);
    }

    #[test]
    fn minmax_on_reverse_unique_slice() {
        let xs: Vec<u8> = vec![5, 4, 3, 2, 1, 0];
        let (a, b) = minmax(&xs);

        assert_eq!(a, 0);
        assert_eq!(b, 5);
    }

    #[test]
    fn minmax_on_repeated() {
        let xs: Vec<u8> = vec![1, 1, 1, 1];
        let (a, b) = minmax(&xs);

        assert_eq!(a, 1);
        assert_eq!(b, 1);
    }

    // logsumexp
    // ---------
    #[test]
    fn logsumexp_on_vector_of_zeros() {
        let xs: Vec<f64> = vec![0.0; 5];
        // should be about log(5)
        assert_relative_eq!(logsumexp(&xs), 1.6094379124341003, epsilon = TOL);
    }

    #[test]
    fn logsumexp_on_random_values() {
        let xs: Vec<f64> = vec![
            0.30415386,
            -0.07072296,
            -1.04287019,
            0.27855407,
            -0.81896765,
        ];
        assert_relative_eq!(logsumexp(&xs), 1.4820007894263059, epsilon = TOL);
    }

    #[test]
    fn logsumexp_returns_only_value_on_one_element_container() {
        let xs: Vec<f64> = vec![0.30415386];
        assert_relative_eq!(logsumexp(&xs), 0.30415386, epsilon = TOL);
    }

    #[test]
    #[should_panic]
    fn logsumexp_should_panic_on_empty() {
        let xs: Vec<f64> = Vec::new();
        logsumexp(&xs);
    }

    // massflip
    // --------
    #[test]
    fn massflip_should_return_valid_indices() {
        let mut rng = ChaChaRng::from_entropy();
        let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 50];
        let ixs = massflip(log_weights, &mut rng);
        assert!(ixs.iter().all(|&ix| ix < 5));
    }

    // bincount
    #[test]
    fn bincount_should_count_occupied() {
        let xs: Vec<u8> = vec![0, 0, 0, 1, 1, 2, 3];
        let counts = bincount(&xs, 4);

        assert_eq!(counts.len(), 4);
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 1);
        assert_eq!(counts[3], 1);
    }

    #[test]
    fn bincount_should_count_with_zeros() {
        let xs: Vec<u8> = vec![0, 0, 0, 2, 2, 2, 3];
        let counts = bincount(&xs, 4);

        assert_eq!(counts.len(), 4);
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 0);
        assert_eq!(counts[2], 3);
        assert_eq!(counts[3], 1);
    }

    #[test]
    fn unused_components_none_unused_should_return_empty() {
        let asgn_vec: Vec<usize> = vec![0, 1, 2, 3, 3, 4];
        let k = 5;
        let unused = unused_components(k, &asgn_vec);
        assert!(unused.is_empty());
    }

    #[test]
    fn unused_components_should_return_unused_indices_in_descending_order() {
        let asgn_vec: Vec<usize> = vec![0, 2, 4];
        let k = 5;
        let unused = unused_components(k, &asgn_vec);
        assert_eq!(unused[0], 3);
        assert_eq!(unused[1], 1);
    }

    #[test]
    fn n_unique_should_work_no_unique() {
        let xs: Vec<f64> = vec![1.3, 1.3, 1.3, 1.3, 1.3];
        let u = n_unique(&xs, 100);
        assert_eq!(u, 1)
    }

    #[test]
    fn n_unique_should_work_all_unique() {
        let xs: Vec<f64> = vec![1.3, 1.4, 2.3, 1.5, 1.6];
        let u = n_unique(&xs, 100);
        assert_eq!(u, 5)
    }

    #[test]
    fn n_unique_should_work_some_unique() {
        let xs: Vec<f64> = vec![1.3, 1.4, 1.3, 1.4, 1.3];
        let u = n_unique(&xs, 100);
        assert_eq!(u, 2)
    }

    #[test]
    fn n_unique_should_max_out_at_cutoff_plus_one() {
        let xs: Vec<f64> = vec![1.2, 1.3, 1.4, 1.5, 1.3];
        let u = n_unique(&xs, 2);
        assert_eq!(u, 3)
    }

    #[test]
    fn tanspose_mapvec() {
        let mut m1: BTreeMap<String, usize> = BTreeMap::new();
        m1.insert(String::from("x"), 1);
        m1.insert(String::from("y"), 2);

        let mut m2: BTreeMap<String, usize> = BTreeMap::new();
        m2.insert(String::from("x"), 3);
        m2.insert(String::from("y"), 4);

        let mut m3: BTreeMap<String, usize> = BTreeMap::new();
        m3.insert(String::from("x"), 5);
        m3.insert(String::from("y"), 6);

        let mapvec = vec![m1, m2, m3];

        let vecmap = transpose_mapvec(&mapvec);

        assert_eq!(vecmap.len(), 2);
        assert_eq!(vecmap[&String::from("x")], vec![1, 3, 5]);
        assert_eq!(vecmap[&String::from("y")], vec![2, 4, 6]);
    }

    #[test]
    fn partition_iterator_creates_right_number_of_partitions() {
        // https://en.wikipedia.org/wiki/Bell_number
        let bell_nums: Vec<(usize, u64)> =
            vec![(1, 1), (2, 2), (3, 5), (4, 15), (5, 52), (6, 203)];

        for (n, bell) in bell_nums {
            let mut count: u64 = 0;
            let parts = Partition::new(n).for_each(|_| count += 1);
            assert_eq!(count, bell);
        }
    }

    #[test]
    fn binom_nk() {
        assert_eq!(binom(5, 0), 1);
        assert_eq!(binom(5, 1), 5);
        assert_eq!(binom(5, 2), 10);
        assert_eq!(binom(5, 3), 10);
        assert_eq!(binom(5, 4), 5);
        assert_eq!(binom(5, 1), 5);

        assert_eq!(binom(10, 6), 210);
        assert_eq!(binom(10, 4), 210);
    }

    #[test]
    fn sterling_nk() {
        assert_eq!(sterling(0, 0), 1);

        assert_eq!(sterling(1, 0), 0);
        assert_eq!(sterling(1, 1), 1);

        assert_eq!(sterling(10, 3), 9330);
        assert_eq!(sterling(10, 4), 34105);
    }

    #[test]
    fn bell_n() {
        assert_eq!(bell(0), 1);
        assert_eq!(bell(1), 1);
        assert_eq!(bell(2), 2);
        assert_eq!(bell(3), 5);
        assert_eq!(bell(4), 15);
        assert_eq!(bell(5), 52);
    }
}

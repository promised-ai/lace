extern crate rand;

use std::collections::{BTreeMap, HashSet};
use std::f64::NAN;
use std::iter::FromIterator;
use std::mem::swap;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use rayon::prelude::*;
use std::cmp::PartialOrd;
use std::ops::AddAssign;
use std::str::FromStr;

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

pub fn var(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    let m = mean(xs);
    let v = xs.iter().fold(0.0, |acc, x| acc + (x - m) * (x - m));
    // TODO: Add dof and return 0 if n == 1
    v / n
}

pub fn mean(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    xs.iter().fold(0.0, |acc, x| x + acc) / n
}

pub fn std(xs: &[f64]) -> f64 {
    let v: f64 = var(xs);
    v.sqrt()
}

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

pub fn choose2ixs<R: Rng>(n: usize, mut rng: &mut R) -> (usize, usize) {
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

pub fn pflip(weights: &[f64], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    if weights.is_empty() {
        panic!("Empty container");
    }
    let ws: Vec<f64> = cumsum(weights);
    let scale: f64 = *ws.last().unwrap();
    let u = Uniform::new(0.0, 1.0);

    (0..n)
        .map(|_| {
            let r = rng.sample(u) * scale;
            match ws.iter().position(|&w| w > r) {
                Some(ix) => ix,
                None => {
                    let wsvec = weights.to_vec();
                    panic!("Could not draw from {:?}", wsvec)
                }
            }
        }).collect()
}

pub fn log_pflip(log_weights: &[f64], rng: &mut impl Rng) -> usize {
    let maxval = *log_weights
        .iter()
        .max_by(|x, y| x.partial_cmp(y).expect(&format!("{:?}", log_weights)))
        .unwrap();
    let mut weights: Vec<f64> =
        log_weights.iter().map(|w| (w - maxval).exp()).collect();

    // doing this instead of calling pflip shaves about 30% off the runtime.
    for i in 1..weights.len() {
        weights[i] += weights[i - 1];
    }

    let scale = *weights.last().unwrap();
    let u = Uniform::new(0.0, scale);
    let r = rng.sample(u);

    match weights.iter().position(|&w| w > r) {
        Some(ix) => ix,
        None => panic!("Could not draw from {:?}", weights),
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
            let maxval =
                *lps.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
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
        }).collect_into_vec(&mut out);
    out
}

pub fn massflip(mut logps: Vec<Vec<f64>>, rng: &mut impl Rng) -> Vec<usize> {
    let k = logps[0].len();
    let mut ixs: Vec<usize> = Vec::with_capacity(logps.len());
    let u = Uniform::new(0.0, 1.0);

    for lps in &mut logps {
        // ixs.push(log_pflip(&lps, &mut rng)); // debug
        let maxval: f64 =
            *lps.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
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
        asgn: asgn,
        counts: counts,
        ncats: ncats,
    }
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

    // pflip
    // -----
    #[test]
    fn pflip_should_always_return_an_index_for_normed_ps() {
        let mut rng = ChaChaRng::from_entropy();
        let weights: Vec<f64> = vec![0.1, 0.2, 0.5, 0.2];
        for _ in 0..100 {
            let ix: usize = pflip(&weights, 1, &mut rng)[0];
            assert!(ix < 4);
        }
    }

    #[test]
    fn pflip_should_always_return_an_index_for_unnormed_ps() {
        let mut rng = ChaChaRng::from_entropy();
        let weights: Vec<f64> = vec![1.0, 2.0, 5.0, 3.5];
        for _ in 0..100 {
            let ix: usize = pflip(&weights, 1, &mut rng)[0];
            assert!(ix < 4);
        }
    }

    #[test]
    fn pflip_should_always_return_zero_for_singluar_array() {
        let mut rng = ChaChaRng::from_entropy();
        for _ in 0..100 {
            let weights: Vec<f64> = vec![0.5];
            let ix: usize = pflip(&weights, 1, &mut rng)[0];
            assert_eq!(ix, 0);
        }
    }

    #[test]
    fn pflip_should_return_draws_in_accordance_with_weights() {
        let mut rng = ChaChaRng::from_entropy();
        let weights: Vec<f64> = vec![0.0, 0.2, 0.5, 0.3];
        let mut counts: Vec<f64> = vec![0.0; 4];
        for _ in 0..10_000 {
            let ix: usize = pflip(&weights, 1, &mut rng)[0];
            counts[ix] += 1.0;
        }
        let ps: Vec<f64> = counts.iter().map(|&x| x / 10_000.0).collect();

        // This might fail sometimes
        assert_relative_eq!(ps[0], 0.0, epsilon = TOL);
        assert_relative_eq!(ps[1], 0.2, epsilon = 0.05);
        assert_relative_eq!(ps[2], 0.5, epsilon = 0.05);
        assert_relative_eq!(ps[3], 0.3, epsilon = 0.05);
    }

    #[test]
    #[should_panic]
    fn pflip_should_panic_given_empty_container() {
        let mut rng = ChaChaRng::from_entropy();
        let weights: Vec<f64> = Vec::new();
        pflip(&weights, 1, &mut rng);
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
}

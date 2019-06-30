use std::collections::{BTreeMap, HashSet};
use std::f64::NAN;
use std::iter::FromIterator;
use std::mem::swap;
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

/// Bins the entries in `xs` into `k` bins.
///
/// # Example
///
/// ```rust
/// # use braid_utils::misc::bincount;
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

// TODO: This is not optimized. If we compare pairs of element, we get 1.5n
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
        let maxval_res = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap());
        let maxval = match maxval_res {
            Some(val) => val,
            None => panic!("Could not find maxval of {:?}", xs),
        };
        xs.iter()
            .fold(0.0_f64, |acc, x| acc + (x - maxval).exp())
            .ln()
            + maxval
    }
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

#[cfg(test)]
mod tests {
    extern crate approx;

    use super::*;
    use approx::*;

    const TOL: f64 = 1E-10;

    // FIXME: parse_result test
    // FIXME: sign test

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

    // bincount
    // --------
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

    // FIXME: transpose test

    // transpose mapvec
    // ----------------
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

    // unused components
    // -----------------
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
}

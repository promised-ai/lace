use std::collections::{BTreeMap, HashSet};
use std::f64::NAN;
use std::mem::swap;
use std::ops::AddAssign;
use std::str::FromStr;

pub fn is_index_col(col_name: &str) -> bool {
    let lower = col_name.to_lowercase();
    lower == "id" || lower == "index"
}

pub trait MinMax {
    type Inner: PartialOrd;
    /// Simultaneously compute the min and max of items in an Iterator. Returns
    /// `None` if the iterator is empty.
    fn minmax(&mut self) -> Option<(Self::Inner, Self::Inner)>;
}

// TODO: This is not optimized. If we compare pairs of elements, we get 1.5n
// comparisons instead of 2n.
impl<T> MinMax for T
where
    T: Iterator,
    T::Item: PartialOrd + Clone,
{
    type Inner = T::Item;
    fn minmax(&mut self) -> Option<(Self::Inner, Self::Inner)> {
        let mut min = self.next()?;

        let mut max = if let Some(item) = self.next() {
            item
        } else {
            return Some((min.clone(), min));
        };

        if min > max {
            swap(&mut min, &mut max);
        }

        for item in self {
            if item > max {
                max = item;
            } else if item < min {
                min = item;
            }
        }
        Some((min, max))
    }
}

/// Attempt to turn a `&str` into a `T`
#[inline]
pub fn parse_result<T: FromStr>(x: &str) -> Result<Option<T>, T::Err> {
    // For csv, empty cells are considered missing regardless of type
    if x.is_empty() {
        Ok(None)
    } else {
        x.parse::<T>().map(Some)
    }
}

/// Like `signum`, but return 0.0 if the number is zero
#[inline]
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
/// # use lace_utils::bincount;
/// let xs: Vec<usize> = vec![0, 0, 1, 2, 2, 2, 3];
///
/// assert_eq!(bincount(&xs, 4), vec![2, 1, 3, 1]);
/// ```
#[inline]
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
#[inline]
pub fn cumsum<T>(xs: &[T]) -> Vec<T>
where
    T: AddAssign + Clone,
{
    let mut summed: Vec<T> = xs.to_vec();
    for i in 1..xs.len() {
        let l = summed[i - 1].clone();
        summed[i] += l;
    }
    summed
}

/// Returns the index of the largest element in xs.
///
/// If there are multiple largest elements, returns the index of the first.
#[inline]
pub fn argmax<T: PartialOrd>(xs: &[T]) -> usize {
    assert!(!xs.is_empty(), "Empty container");

    if xs.len() == 1 {
        0
    } else {
        let (max_ix, _) = xs.iter().enumerate().skip(1).fold(
            (0, &xs[0]),
            |(max_ix, max_val), (ix, x)| {
                if x > max_val {
                    (ix, x)
                } else {
                    (max_ix, max_val)
                }
            },
        );
        max_ix
    }
}

/// Returns the index of the smallest element in xs.
///
/// If there are multiple smallest elements, returns the index of the first.
#[inline]
pub fn argmin<T: PartialOrd>(xs: &[T]) -> usize {
    assert!(!xs.is_empty(), "Empty container");

    if xs.len() == 1 {
        0
    } else {
        let (min_ix, _) = xs.iter().enumerate().skip(1).fold(
            (0, &xs[0]),
            |(min_ix, min_val), (ix, x)| {
                if x < min_val {
                    (ix, x)
                } else {
                    (min_ix, min_val)
                }
            },
        );
        min_ix
    }
}

/// Returns a tuple (min_elem, max_elem).
///
/// Faster than calling min and max individually
#[inline]
pub fn minmax<T: PartialOrd + Clone>(xs: &[T]) -> (T, T) {
    xs.iter().cloned().minmax().expect("Empty slice")
}

/// Numerically stable `log(sum(exp(xs))`
#[inline]
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

/// Perform ln(exp(x) + exp(y)) in a more numerically stable way
///
/// # Examples
///
/// Is equivalent to `logsumexp(&vec![x, y])
///
/// ```
/// # use lace_utils::{logaddexp, logsumexp};
/// let x = -0.00231;
/// let y = -0.08484;
///
/// let lse = logsumexp(&vec![x, y]);
/// let lae = logaddexp(x, y);
///
/// assert!((lse - lae).abs() < 1E-13);
/// ```
#[inline]
pub fn logaddexp(x: f64, y: f64) -> f64 {
    if x > y {
        (y - x).exp().ln_1p() + x
    } else {
        (x - y).exp().ln_1p() + y
    }
}

pub fn transpose<T: Copy + Default>(mat_in: &[Vec<T>]) -> Vec<Vec<T>> {
    let n_rows = mat_in.len();
    let n_cols = mat_in[0].len();
    let mut mat_out: Vec<Vec<T>> = vec![vec![T::default(); n_rows]; n_cols];

    for (i, row) in mat_in.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            mat_out[j][i] = x;
        }
    }

    mat_out
}

/// Turn `Vec<Map<K, V>>` into `Map<K, Vec<V>>`
pub fn transpose_mapvec<K: Clone + Ord, V: Clone>(
    mapvec: &[BTreeMap<K, V>],
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
    let all_cpnts: HashSet<usize> = (0..k).collect();
    let used_cpnts: HashSet<usize> = asgn_vec.iter().cloned().collect();
    let mut unused_cpnts: Vec<usize> =
        all_cpnts.difference(&used_cpnts).cloned().collect();
    unused_cpnts.sort_unstable();
    // needs to be in reverse order, because we want to remove the
    // higher-indexed views first to minimize bookkeeping.
    unused_cpnts.reverse();
    unused_cpnts
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    const TOL: f64 = 1E-10;

    // parse_result
    // ------------
    #[test]
    fn parse_result_f64() {
        {
            let res: Option<f64> = parse_result("1.23").unwrap();
            assert!(res.unwrap() == 1.23);
        }
        {
            let res: Option<f64> = parse_result(".23").unwrap();
            assert!(res.unwrap() == 0.23);
        }
    }

    #[test]
    fn parse_result_u8() {
        {
            let res: Option<u8> = parse_result("1").unwrap();
            assert_eq!(res.unwrap(), 1);
        }
        {
            let res: Option<u8> = parse_result("82").unwrap();
            assert_eq!(res.unwrap(), 82);
        }
    }

    #[test]
    #[should_panic]
    fn parse_result_u8_too_large_fail() {
        let _res: Option<u8> = parse_result("256").unwrap();
    }

    #[test]
    fn parse_empty_is_none() {
        let res: Option<u8> = parse_result("").unwrap();
        assert!(res.is_none());
    }

    // sign
    // ----
    macro_rules! sign_test {
        ($value: expr, $target: expr, $test_name: ident) => {
            #[test]
            fn $test_name() {
                assert_eq!(sign($value), $target);
            }
        };
    }

    sign_test!(-2.5, -1.0, neg_sign_is_neg);
    sign_test!(-1E-14, -1.0, small_neg_sign_is_neg);
    sign_test!(0.0, 0.0, zero_sign_is_zero);
    sign_test!(10.0, 1.0, pos_sign_is_pos);
    sign_test!(1E-14, 1.0, small_pos_sign_is_pos);

    #[test]
    fn nan_sign_is_nan() {
        assert!(sign(NAN).is_nan())
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
        assert_relative_eq!(
            logsumexp(&xs),
            1.609_437_912_434_100_3,
            epsilon = TOL
        );
    }

    #[test]
    fn logsumexp_on_random_values() {
        let xs: Vec<f64> = vec![
            0.304_153_86,
            -0.070_722_96,
            -1.042_870_19,
            0.278_554_07,
            -0.818_967_65,
        ];
        assert_relative_eq!(
            logsumexp(&xs),
            1.482_000_789_426_305_9,
            epsilon = TOL
        );
    }

    #[test]
    fn logsumexp_returns_only_value_on_one_element_container() {
        let xs: Vec<f64> = vec![0.304_153_86];
        assert_relative_eq!(logsumexp(&xs), 0.304_153_86, epsilon = TOL);
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

    // transpose
    // ---------
    #[test]
    fn transpose_square() {
        let xs = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];

        let xt = transpose(&xs);

        assert_eq!(xt, vec![vec![0, 3, 6], vec![1, 4, 7], vec![2, 5, 8],],);
    }

    #[test]
    fn transpose_rect() {
        let xs = vec![vec![0, 1, 2], vec![3, 4, 5]];

        let xt = transpose(&xs);

        assert_eq!(xt, vec![vec![0, 3], vec![1, 4], vec![2, 5],],);
    }

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

    #[test]
    fn is_index_col_tests() {
        assert!(is_index_col("ID"));
        assert!(is_index_col("id"));
        assert!(is_index_col("iD"));
        assert!(is_index_col("Id"));
        assert!(is_index_col("Index"));
        assert!(is_index_col("index"));

        assert!(!is_index_col("idindex"));
        assert!(!is_index_col("indexid"));
        assert!(!is_index_col(""));
        assert!(!is_index_col("icecream"));
    }
}

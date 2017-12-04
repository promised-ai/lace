extern crate rand;
pub mod mh;

use std::iter::FromIterator;
use std::collections::HashSet;
use rayon::prelude::*;
use std::ops::AddAssign;
use self::rand::Rng;
use std::f64::NEG_INFINITY;
use std::cmp::PartialOrd;


pub fn var(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    let m = mean(xs);
    let v = xs.iter().fold(0.0, |acc, x| acc + (x - m)*(x - m));
    v / n
}


pub fn mean(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    xs.iter().fold(0.0, |acc, x| x + acc) / n
}


pub fn bincount<T>(xs: &[T], k: usize) -> Vec<usize>
    where T: Clone + Into<usize>
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
    where T: AddAssign + Clone
{
    let mut summed: Vec<T> = xs.to_vec();
    for i in 1..xs.len() {
        summed[i] += summed[i - 1].clone();
    }
    summed
}


pub fn argmax(xs: &[f64]) -> usize {
    if xs.is_empty() {
        panic!("Empty container");
    }
    let mut maxval = NEG_INFINITY;
    let mut max_ix: usize = 0;
    for (ix, &x) in xs.iter().enumerate() {
        if x > maxval {
            maxval = x;
            max_ix = ix;
        }
    }
    max_ix
}


pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        panic!("Empty container");
    } else if xs.len() == 1 {
        xs[0]
    } else {
        let maxval = *xs.iter()
                        .max_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap();
        xs.iter().fold(0.0, |acc, x| acc + (x - maxval).exp()).ln() + maxval
    }
}


pub fn pflip(weights: &[f64], rng: &mut Rng) -> usize {
    if weights.is_empty() {
        panic!("Empty container");
    }
    let ws: Vec<f64> = cumsum(weights);
    let scale: f64 = *ws.last().unwrap();
    let r = rng.next_f64() * scale;

    match ws.iter().position(|&w| w > r) {
        Some(ix) => ix,
        None     => {
            let wsvec = weights.to_vec();
            panic!("Could not draw from {:?}", wsvec)},
    }

}


pub fn log_pflip(log_weights: &[f64], rng: &mut Rng) -> usize {
    let maxval = *log_weights.iter()
                             .max_by(|x, y| x.partial_cmp(y).unwrap())
                             .unwrap(); 
    let mut weights: Vec<f64> = log_weights.iter()
                                           .map(|w| (w-maxval).exp())
                                           .collect();

    // doing this instead of calling pflip shaves about 30% off the runtime.
    for i in 1..weights.len() {
        weights[i] += weights[i-1];
    }

    let scale = *weights.last().unwrap();
    let r = rng.next_f64() * scale;

    match weights.iter().position(|&w| w > r) {
        Some(ix) => ix,
        None     => {panic!("Could not draw from {:?}", weights)},
    }
}


pub fn massflip_par<R: Rng>(mut logps: Vec<Vec<f64>>,
                            rng: &mut R) -> Vec<usize> {
    let n = logps.len();
    let k = logps[0].len();
    let us: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

    let mut out: Vec<usize> = Vec::with_capacity(n);
    logps.par_iter_mut().zip_eq(us.par_iter()).map(|(lps, u)| {
        let maxval = *lps.iter()
                         .max_by(|x, y| x.partial_cmp(y).unwrap())
                         .unwrap(); 
        lps[0] -= maxval;
        lps[0] = lps[0].exp();
        for i in 1..k {
            lps[i] -= maxval;
            lps[i] = lps[i].exp();
            lps[i] += lps[i-1]
        }

        let r = u * *lps.last().unwrap();

        // Is a for loop faster?
        lps.iter().fold(0, |acc, &p| acc + ((p < r) as usize))
    }).collect_into(&mut out);
    out
}


pub fn massflip<R: Rng>(mut logps: Vec<Vec<f64>>, rng: &mut R) -> Vec<usize>
{
    let k = logps[0].len();
    let mut ixs: Vec<usize> = Vec::with_capacity(logps.len());

    for lps in &mut logps {
        // ixs.push(log_pflip(&lps, &mut rng)); // debug
        let maxval: f64 = *lps.iter()
                              .max_by(|x, y| x.partial_cmp(y).unwrap())
                              .unwrap(); 
        lps[0] -= maxval;
        lps[0] = lps[0].exp();
        for i in 1..k {
            lps[i] -= maxval;
            lps[i] = lps[i].exp();
            lps[i] += lps[i-1]
        }

        let scale: f64 = *lps.last().unwrap();
        let r: f64 = rng.next_f64() * scale;

        let mut ct: usize = 0;
        for p in lps {
            ct += (*p < r) as usize;
        }
        ixs.push(ct);
    }
    ixs
}


pub fn massflip_flat<R: Rng>(mut logps: Vec<f64>, n: usize, k: usize,
                             rng: &mut R) -> Vec<usize> {
    let mut ixs: Vec<usize> = Vec::with_capacity(logps.len());
    let mut a = 0;
    while a < n*k {
        let b = a + k - 1;
        let maxval: f64 = *logps[a..b].iter()
                                      .max_by(|x, y| x.partial_cmp(y).unwrap())
                                      .unwrap(); 
        logps[a] -= maxval;
        logps[a] = logps[a].exp();
        for j in a+1..b {
            logps[j] -= maxval;
            logps[j] = logps[j].exp();
            logps[j] += logps[j-1]
        }
        let scale: f64 = logps[b];
        let r: f64 = rng.next_f64() * scale;

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
pub fn unused_components(k: usize, asgn_vec: &[usize]) -> Vec<usize>
{
    let all_cpnts: HashSet<_> = HashSet::from_iter(0..k);
    let used_cpnts = HashSet::from_iter(asgn_vec.iter().cloned());
    let mut unused_cpnts: Vec<&usize> = all_cpnts.difference(&used_cpnts)
                                                  .collect();
    unused_cpnts.sort();
    // needs to be in reverse order, because we want to remove the
    // higher-indexed views first to minimize bookkeeping.
    unused_cpnts.reverse();
    unused_cpnts.iter().map(|&z| *z).collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use self::rand::chacha::ChaChaRng;

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
        let xs: Vec<f64> = vec![1.0/3.0, 2.0/3.0, 5.0/8.0, 11.0/12.0];
        assert_relative_eq!(mean(&xs), 0.63541666666666663, epsilon = 10E-8);
    }

    #[test]
    fn var_1() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(var(&xs), 2.0, epsilon = 10E-10);
    }

    #[test]
    fn var_2() {
        let xs: Vec<f64> = vec![1.0/3.0, 2.0/3.0, 5.0/8.0, 11.0/12.0];
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


    // logsumexp
    // ---------
    #[test]
    fn logsumexp_on_vector_of_zeros(){
        let xs: Vec<f64> = vec![0.0; 5];
        // should be about log(5)
        assert_relative_eq!(logsumexp(&xs), 1.6094379124341003, epsilon = TOL);
    }

    #[test]
    fn logsumexp_on_random_values() {
        let xs: Vec<f64> = vec![0.30415386, -0.07072296, -1.04287019,
                                0.27855407, -0.81896765];
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
    fn pflip_should_always_return_an_index_for_normed_ps(){
        let mut rng = ChaChaRng::new_unseeded();
        let weights: Vec<f64> = vec![0.1, 0.2, 0.5, 0.2];
        for _ in 0..100 {
            let ix: usize = pflip(&weights, &mut rng);
            assert!(ix < 4);
        }
    }

    #[test]
    fn pflip_should_always_return_an_index_for_unnormed_ps(){
        let mut rng = ChaChaRng::new_unseeded();
        let weights: Vec<f64> = vec![1.0, 2.0, 5.0, 3.5];
        for _ in 0..100 {
            let ix: usize = pflip(&weights, &mut rng);
            assert!(ix < 4);
        }
    }

    #[test]
    fn pflip_should_always_return_zero_for_singluar_array() {
        let mut rng = ChaChaRng::new_unseeded();
        for _ in 0..100 {
            let weights: Vec<f64> = vec![0.5];
            let ix: usize = pflip(&weights, &mut rng);
            assert_eq!(ix, 0);
        }
    }

    #[test]
    fn pflip_should_return_draws_in_accordance_with_weights() {
        let mut rng = ChaChaRng::new_unseeded();
        let weights: Vec<f64> = vec![0.0, 0.2, 0.5, 0.3];
        let mut counts: Vec<f64> = vec![0.0; 4];
        for _ in 0..10_000 {
            let ix: usize = pflip(&weights, &mut rng);
            counts[ix] += 1.0;
        }
        let ps: Vec<f64> = counts.iter().map(|&x| x/10_000.0).collect();

        // This might fail sometimes
        assert_relative_eq!(ps[0], 0.0, epsilon = TOL);
        assert_relative_eq!(ps[1], 0.2, epsilon = 0.05);
        assert_relative_eq!(ps[2], 0.5, epsilon = 0.05);
        assert_relative_eq!(ps[3], 0.3, epsilon = 0.05);
    }

    #[test]
    #[should_panic]
    fn pflip_should_panic_given_empty_container() {
        let mut rng = ChaChaRng::new_unseeded();
        let weights: Vec<f64> = Vec::new();
        pflip(&weights, &mut rng);

    }


    // massflip
    // --------
    #[test]
    fn massflip_should_return_valid_indices() {
        let mut rng = ChaChaRng::new_unseeded();
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
}

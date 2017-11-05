#[macro_use] extern crate assert_approx_eq;

extern crate rand;
extern crate braid;

use braid::misc;
use rand::chacha::ChaChaRng;

const TOL: f64 = 1E-10;

// minf64
// ------
#[test]
fn minf64_should_find_min_of_unique_values() {
    let xs: Vec<f64> = vec![0.0, 1.0, 2.0, -1.0];
    assert_approx_eq!(-1.0, misc::minf64(&xs), TOL);
}

#[test]
fn minf64_should_find_min_of_repeat_values() {
    let xs: Vec<f64> = vec![0.0, -2.0, 2.0, -2.0];
    assert_approx_eq!(-2.0, misc::minf64(&xs), TOL);
}

#[test]
fn minf64_should_find_min_of_identical_values() {
    let xs: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
    assert_approx_eq!(1.0, misc::minf64(&xs), TOL);
}

#[test]
fn minf64_should_return_only_value_in_one_length_container() {
    let xs: Vec<f64> = vec![2.0];
    assert_approx_eq!(2.0, misc::minf64(&xs), TOL);
}

#[test]
#[should_panic]
fn minf64_should_panic_on_empty_vec() {
    let xs: Vec<f64> = Vec::new();
    misc::minf64(&xs);
}


// cumsum
// ------
#[test]
fn cumsum_should_work_on_u8() {
    let xs: Vec<u8> = vec![2, 3, 4, 1, 0];
    assert_eq!(misc::cumsum(&xs), [2, 5, 9, 10, 10]);
}

#[test]
fn cumsum_should_work_on_u16() {
    let xs: Vec<u16> = vec![2, 3, 4, 1, 0];
    assert_eq!(misc::cumsum(&xs), [2, 5, 9, 10, 10]);
}

#[test]
fn cumsum_should_work_on_f64() {
    let xs: Vec<f64> = vec![2.0, 3.0, 4.0, 1.0, 0.1];
    assert_eq!(misc::cumsum(&xs), [2.0, 5.0, 9.0, 10.0, 10.1]);
}

#[test]
fn cumsum_should_work_do_nothing_to_one_length_vector() {
    let xs: Vec<u8> = vec![2];
    assert_eq!(misc::cumsum(&xs), [2]);
}

#[test]
fn cumsum_should_return_empty_if_given_empty() {
    let xs: Vec<f64> = Vec::new();
    assert!(misc::cumsum(&xs).is_empty());
}


// argmax
// ------
#[test]
fn argmax_should_work_on_unique_values() {
    let xs: Vec<f64> = vec![2.0, 3.0, 4.0, 1.0, 0.1];
    assert_eq!(misc::argmax(&xs), 2);
}

#[test]
fn argmax_should_return_0_if_max_value_is_in_0_index() {
    let xs: Vec<f64> = vec![20.0, 3.0, 4.0, 1.0, 0.1];
    assert_eq!(misc::argmax(&xs), 0);
}

#[test]
fn argmax_should_return_last_index_if_max_value_is_last() {
    let xs: Vec<f64> = vec![0.0, 3.0, 4.0, 1.0, 20.1];
    assert_eq!(misc::argmax(&xs), 4);
}

#[test]
fn argmax_should_return_index_of_first_max_value_if_repeats() {
    let xs: Vec<f64> = vec![0.0, 0.0, 2.0, 1.0, 2.0];
    assert_eq!(misc::argmax(&xs), 2);
}

#[test]
#[should_panic]
fn argmax_should_panic_given_empty_container() {
    let xs: Vec<f64> = Vec::new();
    misc::argmax(&xs);
}


// logsumexp
// ---------
#[test]
fn logsumexp_on_vector_of_zeros(){
    let xs: Vec<f64> = vec![0.0; 5];
    // should be about log(5)
    assert_approx_eq!(misc::logsumexp(&xs), 1.6094379124341003, TOL);
}

#[test]
fn logsumexp_on_random_values() {
    let xs: Vec<f64> = vec![0.30415386, -0.07072296, -1.04287019, 0.27855407, -0.81896765];
    assert_approx_eq!(misc::logsumexp(&xs), 1.4820007894263059, TOL);
}

#[test]
fn logsumexp_returns_only_value_on_one_element_container() {
    let xs: Vec<f64> = vec![0.30415386];
    assert_approx_eq!(misc::logsumexp(&xs), 0.30415386, TOL);
}

#[test]
#[should_panic]
fn logsumexp_should_panic_on_empty() {
    let xs: Vec<f64> = Vec::new();
    misc::logsumexp(&xs);
}


// pflip
// -----
#[test]
fn pflip_should_always_return_an_index_for_normed_ps(){
    let mut rng = ChaChaRng::new_unseeded();
    let weights: Vec<f64> = vec![0.1, 0.2, 0.5, 0.2];
    for _ in 0..100 {
        let ix: usize = misc::pflip(&weights, &mut rng);
        assert!(ix < 4);
    }
}

#[test]
fn pflip_should_always_return_an_index_for_unnormed_ps(){
    let mut rng = ChaChaRng::new_unseeded();
    let weights: Vec<f64> = vec![1.0, 2.0, 5.0, 3.5];
    for _ in 0..100 {
        let ix: usize = misc::pflip(&weights, &mut rng);
        assert!(ix < 4);
    }
}

#[test]
fn pflip_should_always_return_zero_for_singluar_array() {
    let mut rng = ChaChaRng::new_unseeded();
    for _ in 0..100 {
        let weights: Vec<f64> = vec![0.5];
        let ix: usize = misc::pflip(&weights, &mut rng);
        assert_eq!(ix, 0);
    }
}

#[test]
fn pflip_should_return_draws_in_accordance_with_weights() {
    let mut rng = ChaChaRng::new_unseeded();
    let weights: Vec<f64> = vec![0.0, 0.2, 0.5, 0.3];
    let mut counts: Vec<f64> = vec![0.0; 4];
    for _ in 0..10_000 {
        let ix: usize = misc::pflip(&weights, &mut rng);
        counts[ix] += 1.0;
    }
    let ps: Vec<f64> = counts.iter().map(|&x| x/10_000.0).collect();

    // This might fail sometimes
    assert_approx_eq!(ps[0], 0.0, TOL);
    assert_approx_eq!(ps[1], 0.2, 0.05);
    assert_approx_eq!(ps[2], 0.5, 0.05);
    assert_approx_eq!(ps[3], 0.3, 0.05);
}

#[test]
#[should_panic]
fn pflip_should_panic_given_empty_container() {
    let mut rng = ChaChaRng::new_unseeded();
    let weights: Vec<f64> = Vec::new(); 
    misc::pflip(&weights, &mut rng);

}


// massflip
// --------
#[test]
fn massflip_should_return_valid_indices() {
    let mut rng = ChaChaRng::new_unseeded();
    let log_weights: Vec<Vec<f64>> = vec![vec![0.0; 5]; 50];
    let ixs = misc::massflip(log_weights, &mut rng);
    assert!(ixs.iter().all(|&ix| ix < 5));
}


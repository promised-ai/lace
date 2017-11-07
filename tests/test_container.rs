#[macro_use] extern crate assert_approx_eq;

extern crate braid;

use braid::cc::DataContainer;
use std::f64::NAN;

#[test]
fn default_container_f64_should_all_construct_properly() {
    let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let container = DataContainer::new(data);

    assert_eq!(container.data.len(), 4);
    assert_eq!(container.present.len(), 4);

    assert!(container.present.iter().all(|&x| x));

    assert_approx_eq!(container.data[0], 0.0, 1e-10);
    assert_approx_eq!(container.data[1], 1.0, 1e-10);
    assert_approx_eq!(container.data[2], 2.0, 1e-10);
    assert_approx_eq!(container.data[3], 3.0, 1e-10);
}


#[test]
fn default_container_u8_should_all_construct_properly() {
    let data: Vec<u8> = vec![0, 1, 2, 3];
    let container = DataContainer::new(data);

    assert_eq!(container.data.len(), 4);
    assert_eq!(container.present.len(), 4);

    assert!(container.present.iter().all(|&x| x));

    assert_eq!(container.data[0], 0);
    assert_eq!(container.data[1], 1);
    assert_eq!(container.data[2], 2);
    assert_eq!(container.data[3], 3);
}


#[test]
fn default_container_bool_should_all_construct_properly() {
    let data: Vec<bool> = vec![true, false, false, true];
    let container = DataContainer::new(data);

    assert_eq!(container.data.len(), 4);
    assert_eq!(container.present.len(), 4);

    assert!(container.present.iter().all(|&x| x));

    assert_eq!(container.data[0], true);
    assert_eq!(container.data[1], false);
    assert_eq!(container.data[2], false);
    assert_eq!(container.data[3], true);
}


#[test]
fn test_index_impl() {
    let data: Vec<u8> = vec![0, 1, 2, 3];
    let container = DataContainer::new(data);

    assert_eq!(container[0], 0);
    assert_eq!(container[1], 1);
    assert_eq!(container[2], 2);
    assert_eq!(container[3], 3);
}


#[test]
fn test_index_mut_impl() {
    let data: Vec<u8> = vec![0, 1, 2, 3];
    let mut container = DataContainer::new(data);

    assert_eq!(container[0], 0);
    assert_eq!(container[1], 1);
    assert_eq!(container[2], 2);
    assert_eq!(container[3], 3);

    container[2] = 97;

    assert_eq!(container[0], 0);
    assert_eq!(container[1], 1);
    assert_eq!(container[2], 97);
    assert_eq!(container[3], 3);
}


#[test]
fn filter_container_u8_should_tag_and_set_missing_values() {
    let data: Vec<u8> = vec![0, 1, 99, 3];

    // the filter identifies present (non-missing) values
    let container = DataContainer::with_filter(data, 0, |&x| x != 99);

    assert!(container.present[0]);
    assert!(container.present[1]);
    assert!(!container.present[2]);
    assert!(container.present[3]);

    assert_eq!(container[0], 0);
    assert_eq!(container[1], 1);
    assert_eq!(container[2], 0);
    assert_eq!(container[3], 3);
}


#[test]
fn filter_container_f64_nan_should_tag_and_set_missing_values() {
    let data: Vec<f64> = vec![0.0, 1.0, NAN, 3.0];

    // the filter identifies present (non-missing) values
    let container = DataContainer::with_filter(data, 0.0, |&x| x.is_finite());

    assert!(container.present[0]);
    assert!(container.present[1]);
    assert!(!container.present[2]);
    assert!(container.present[3]);

    assert_approx_eq!(container[0], 0.0, 1E-10);
    assert_approx_eq!(container[1], 1.0, 1E-10);
    assert_approx_eq!(container[2], 0.0, 1E-10);
    assert_approx_eq!(container[3], 3.0, 1E-10);
}

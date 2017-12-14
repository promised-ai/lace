#![feature(test)]

extern crate braid;
extern crate rand;
extern crate test;

use test::Bencher;
use braid::cc::{Teller, DType};


fn get_small_teller_from_yaml() -> Teller {
    let filenames = vec![
        "resources/test/small-state-1.yaml",
        "resources/test/small-state-2.yaml",
        "resources/test/small-state-3.yaml"];

    Teller::from_yaml(filenames)
}


#[bench]
fn simulate_100_single_vales_from_small_state(b: &mut Bencher){
    let teller = get_small_teller_from_yaml();
    let mut rng = rand::thread_rng();
    let col_ixs = vec![0];
    b.iter(|| {
        test::black_box(teller.simulate(&col_ixs, &None, 100, &mut rng));
    });
}


#[bench]
fn mutual_information_100_samples_in_small_state(b: &mut Bencher){
    let teller = get_small_teller_from_yaml();
    let mut rng = rand::thread_rng();
    b.iter(|| {
        test::black_box(teller.mutual_information(0, 1, 100, &mut rng));
    });
}

#[bench]
fn joint_pdf_from_small_state(b: &mut Bencher){
    let teller = get_small_teller_from_yaml();
    let mut rng = rand::thread_rng();
    let col_ixs = vec![0, 1];
    let vals = vec![DType::Continuous(1.2), DType::Continuous(0.3)];
    b.iter(|| {
        test::black_box(teller.logp(&col_ixs, &vals, &None));
    });
}

#![feature(test)]

extern crate braid;
extern crate rand;
extern crate test;

use test::Bencher;
use braid::interface::{Oracle, DType, MiType};


fn get_small_oracle_from_yaml() -> Teller {
    let filenames = vec![
        "resources/test/small-state-1.yaml",
        "resources/test/small-state-2.yaml",
        "resources/test/small-state-3.yaml"];

    Oracle::from_yaml(filenames)
}


#[bench]
fn simulate_100_singletons_from_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    let mut rng = rand::thread_rng();
    let col_ixs = vec![0];
    b.iter(|| {
        test::black_box(teller.simulate(&col_ixs, &None, 100, &mut rng));
    });
}


#[bench]
fn simulate_100_pairs_from_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    let mut rng = rand::thread_rng();
    let col_ixs = vec![0, 1];
    b.iter(|| {
        test::black_box(teller.simulate(&col_ixs, &None, 100, &mut rng));
    });
}


#[bench]
fn mutual_information_100_samples_in_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    let mut rng = rand::thread_rng();
    b.iter(|| {
        test::black_box(
            teller.mutual_information(0, 1, 100, MiType::UnNormed, &mut rng)
            );
    });
}


#[bench]
fn joint_pdf_from_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();

    let col_ixs = vec![0, 1];
    let vals = vec![vec![DType::Continuous(1.2), DType::Continuous(0.3)]];
    b.iter(|| {
        test::black_box(teller.logp(&col_ixs, &vals, &None));
    });
}


#[bench]
fn rowsim_from_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    b.iter(|| {
        test::black_box(teller.rowsim(0, 1, None));
    });
}


#[bench]
fn depprob_from_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    b.iter(|| {
        test::black_box(teller.depprob(0, 1));
    });
}


#[bench]
fn kl_uncertainty_from_small_state(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    let mut rng = rand::thread_rng();
    b.iter(|| {
        test::black_box(teller.predictive_uncertainty(0, 1, 0, &mut rng));
    });
}

#[bench]
fn js_uncertainty_from_small_state_1000_samples(b: &mut Bencher){
    let teller = get_small_oracle_from_yaml();
    let mut rng = rand::thread_rng();
    b.iter(|| {
        test::black_box(teller.predictive_uncertainty(0, 1, 1_000, &mut rng));
    });
}

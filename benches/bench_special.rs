#![feature(test)]

extern crate braid;
extern crate test;

use test::Bencher;
use braid::special;

// gamma
// -----
#[bench]
fn gamma_small_val(b: &mut Bencher) {
    b.iter(|| {
        test::black_box(special::gamma(0.0001));
    });
}

#[bench]
fn gamma_medium_val(b: &mut Bencher) {
    b.iter(|| {
        test::black_box(special::gamma(11.0));
    });
}

#[bench]
fn gamma_large_val(b: &mut Bencher) {
    b.iter(|| {
        test::black_box(special::gamma(13.0));
    });
}


// log gamma
// ---------
#[bench]
fn gammaln_medum_val(b: &mut Bencher) {
    b.iter(|| {
        test::black_box(special::gammaln(11.0));
    });
}

#[bench]
fn gammaln_large_val(b: &mut Bencher) {
    b.iter(|| {
        test::black_box(special::gammaln(13.0));
    });
}


// Error function
// --------------
#[bench]
fn erf_val(b: &mut Bencher) {
    // there is only one flow to the current erf function
    b.iter(|| {
        test::black_box(special::erf(0.25));
    });
}


// Error function
// --------------
#[bench]
fn erfinv_small_val(b: &mut Bencher) {
    // there is only one flow to the current erf function
    b.iter(|| {
        test::black_box(special::erfinv(0.25));
    });
}

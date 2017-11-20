#[macro_use] extern crate approx;

extern crate rand;
extern crate braid;

use std::collections::BTreeMap;

use self::rand::Rng;
use braid::cc::Assignment;
use braid::cc::DataContainer;
use braid::cc::Feature;
use braid::cc::Column;
use braid::cc::View;
use braid::cc::view::RowAssignAlg;
use braid::dist::Gaussian;
use braid::dist::traits::RandomVariate;
use braid::dist::prior::NormalInverseGamma;


type GaussCol = Column<f64, Gaussian, NormalInverseGamma>;


fn gen_col(id: usize, n: usize, mut rng: &mut Rng) -> GaussCol {
    let gauss = Gaussian::new(0.0, 1.0);
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);

    Column::new(id, data, prior)
}


fn gen_gauss_view(n: usize, mut rng: &mut Rng) -> View {
    let mut ftrs: Vec<Box<Feature>> = vec![];
    ftrs.push(Box::new(gen_col(0, n, &mut rng)));
    ftrs.push(Box::new(gen_col(1, n, &mut rng)));
    ftrs.push(Box::new(gen_col(2, n, &mut rng)));
    ftrs.push(Box::new(gen_col(3, n, &mut rng)));

    View::new(ftrs, 1.0, &mut rng)
}

#[test]
fn create_view_smoke() {
    let mut rng = rand::thread_rng();
    let view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.nrows(), 10);
    assert_eq!(view.ncols(), 4);
}


#[test]
fn finite_reassign_direct_call() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    view.reassign_rows_finite_cpu(&mut rng);
    assert!(view.asgn.validate().is_valid());
}


#[test]
fn finite_reassign_from_reassign() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    view.reassign(RowAssignAlg::FiniteCpu, &mut rng);
    assert!(view.asgn.validate().is_valid());
}


#[test]
fn insert_feature() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.ncols(), 4);

    let new_ftr = Box::new(gen_col(4, 10, &mut rng));

    view.insert_feature(new_ftr, &mut rng);

    assert_eq!(view.ncols(), 5);
}

#[test]
#[should_panic]
fn insert_feature_with_existing_id_panics() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.ncols(), 4);

    let new_ftr = Box::new(gen_col(2, 10, &mut rng));

    view.insert_feature(new_ftr, &mut rng);
}


#[test]
fn remove_feature() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.ncols(), 4);

    let ftr_opt = view.remove_feature(2);

    assert!(ftr_opt.is_some());
    assert_eq!(view.ncols(), 3);
    assert_eq!(ftr_opt.unwrap().id(), 2);
}


#[test]
fn remove_non_existent_feature_returns_none() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.ncols(), 4);

    let ftr_opt = view.remove_feature(14);

    assert!(ftr_opt.is_none());
    assert_eq!(view.ncols(), 4);
}

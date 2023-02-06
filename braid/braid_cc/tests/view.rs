use braid_data::SparseContainer;
use braid_stats::prior::nix::NixHyper;
use braid_stats::rv::dist::{Gaussian, NormalInvChiSquared};
use braid_stats::rv::traits::Rv;
use rand::Rng;

use braid_cc::alg::RowAssignAlg;
use braid_cc::feature::{ColModel, Column, Feature};
use braid_cc::view::{Builder, View};

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = SparseContainer::from(data_vec);
    let hyper = NixHyper::default();
    let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);

    let ftr = Column::new(id, data, prior, hyper);
    ColModel::Continuous(ftr)
}

fn gen_gauss_view<R: Rng>(n: usize, mut rng: &mut R) -> View {
    let mut ftrs: Vec<ColModel> = vec![];
    ftrs.push(gen_col(0, n, &mut rng));
    ftrs.push(gen_col(1, n, &mut rng));
    ftrs.push(gen_col(2, n, &mut rng));
    ftrs.push(gen_col(3, n, &mut rng));

    Builder::new(n)
        .features(ftrs)
        .seed_from_rng(&mut rng)
        .build()
}

#[test]
fn create_view_smoke() {
    let mut rng = rand::thread_rng();
    let view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.n_rows(), 10);
    assert_eq!(view.n_cols(), 4);
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

    assert_eq!(view.n_cols(), 4);

    let new_ftr = gen_col(4, 10, &mut rng);

    view.insert_feature(new_ftr, &mut rng);

    assert_eq!(view.n_cols(), 5);
}

#[test]
#[should_panic]
fn insert_feature_with_existing_id_panics() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.n_cols(), 4);

    let new_ftr = gen_col(2, 10, &mut rng);

    view.insert_feature(new_ftr, &mut rng);
}

#[test]
fn remove_feature() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.n_cols(), 4);

    let ftr_opt = view.remove_feature(2);

    assert!(ftr_opt.is_some());
    assert_eq!(view.n_cols(), 3);
    assert_eq!(ftr_opt.unwrap().id(), 2);
}

#[test]
fn remove_non_existent_feature_returns_none() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.n_cols(), 4);

    let ftr_opt = view.remove_feature(14);

    assert!(ftr_opt.is_none());
    assert_eq!(view.n_cols(), 4);
}

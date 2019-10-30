#[macro_use]
extern crate approx;

use braid::cc::AppendRowsData;
use braid::cc::{
    ColModel, Column, DataContainer, Feature, RowAssignAlg, View, ViewBuilder,
};
use braid_stats::prior::{Ng, NigHyper};
use braid_stats::Datum;
use rand::Rng;
use rv::dist::Gaussian;
use rv::traits::Rv;

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let hyper = NigHyper::default();
    let prior = Ng::new(0.0, 1.0, 1.0, 1.0, hyper);

    let ftr = Column::new(id, data, prior);
    ColModel::Continuous(ftr)
}

fn gen_gauss_view<R: Rng>(n: usize, mut rng: &mut R) -> View {
    let mut ftrs: Vec<ColModel> = vec![];
    ftrs.push(gen_col(0, n, &mut rng));
    ftrs.push(gen_col(1, n, &mut rng));
    ftrs.push(gen_col(2, n, &mut rng));
    ftrs.push(gen_col(3, n, &mut rng));

    ViewBuilder::new(n)
        .with_features(ftrs)
        .seed_from_rng(&mut rng)
        .build()
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

    let new_ftr = gen_col(4, 10, &mut rng);

    view.insert_feature(new_ftr, &mut rng);

    assert_eq!(view.ncols(), 5);
}

#[test]
#[should_panic]
fn insert_feature_with_existing_id_panics() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.ncols(), 4);

    let new_ftr = gen_col(2, 10, &mut rng);

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

#[test]
fn append_row_present_ordered() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.nrows(), 10);

    let x_0 = AppendRowsData::new(2, vec![Datum::Continuous(3.3)]);
    let x_1 = AppendRowsData::new(1, vec![Datum::Continuous(2.2)]);
    let x_2 = AppendRowsData::new(3, vec![Datum::Continuous(4.4)]);
    let x_3 = AppendRowsData::new(0, vec![Datum::Continuous(1.1)]);

    let new_row = vec![&x_0, &x_1, &x_2, &x_3];

    view.append_rows(new_row, &mut rng);

    assert_eq!(view.nrows(), 11);
    assert!(view.asgn.validate().is_valid());
}

#[test]
fn append_row_present_unordered() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.nrows(), 10);

    let x_0 = AppendRowsData::new(0, vec![Datum::Continuous(1.1)]);
    let x_1 = AppendRowsData::new(1, vec![Datum::Continuous(2.2)]);
    let x_2 = AppendRowsData::new(2, vec![Datum::Continuous(3.3)]);
    let x_3 = AppendRowsData::new(3, vec![Datum::Continuous(4.4)]);

    let new_row = vec![&x_1, &x_0, &x_3, &x_2];

    view.append_rows(new_row, &mut rng);

    assert_eq!(view.nrows(), 11);
    assert!(view.asgn.validate().is_valid());

    let y_0 = view.datum(10, 0).unwrap().to_f64_opt().unwrap();
    let y_1 = view.datum(10, 1).unwrap().to_f64_opt().unwrap();
    let y_2 = view.datum(10, 2).unwrap().to_f64_opt().unwrap();
    let y_3 = view.datum(10, 3).unwrap().to_f64_opt().unwrap();

    assert_relative_eq!(y_0, 1.1, epsilon = 1E-10);
    assert_relative_eq!(y_1, 2.2, epsilon = 1E-10);
    assert_relative_eq!(y_2, 3.3, epsilon = 1E-10);
    assert_relative_eq!(y_3, 4.4, epsilon = 1E-10);
}

#[test]
fn append_row_partial_unordered() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.nrows(), 10);

    let x_0 = AppendRowsData::new(2, vec![Datum::Missing]);
    let x_1 = AppendRowsData::new(1, vec![Datum::Missing]);
    let x_2 = AppendRowsData::new(3, vec![Datum::Continuous(4.4)]);
    let x_3 = AppendRowsData::new(0, vec![Datum::Continuous(1.1)]);

    let new_row = vec![&x_1, &x_0, &x_3, &x_2];

    view.append_rows(new_row, &mut rng);

    assert_eq!(view.nrows(), 11);
    assert!(view.asgn.validate().is_valid());
}

#[test]
fn append_rows_partial_unordered() {
    let mut rng = rand::thread_rng();
    let mut view = gen_gauss_view(10, &mut rng);

    assert_eq!(view.nrows(), 10);

    let x_0 =
        AppendRowsData::new(2, vec![Datum::Missing, Datum::Continuous(2.2)]);
    let x_1 =
        AppendRowsData::new(1, vec![Datum::Missing, Datum::Continuous(1.1)]);
    let x_2 =
        AppendRowsData::new(3, vec![Datum::Continuous(4.4), Datum::Missing]);
    let x_3 =
        AppendRowsData::new(0, vec![Datum::Continuous(1.1), Datum::Missing]);

    let new_row = vec![&x_1, &x_0, &x_3, &x_2];

    view.append_rows(new_row, &mut rng);

    assert_eq!(view.nrows(), 12);
    assert!(view.asgn.validate().is_valid());
}

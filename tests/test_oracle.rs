use approx::*;

use std::fs::File;
use std::io::Read;
use std::path::Path;

use braid::cc::{ColModel, Column, DataContainer, DataStore, State};
use braid::{Given, Oracle, OracleT};
use braid_codebook::Codebook;
use braid_stats::prior::{Ng, NigHyper};
use rand::Rng;
use rv::dist::{Gamma, Gaussian, Mixture};
use rv::traits::{Cdf, Rv};

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let hyper = NigHyper::default();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = DataContainer::new(data_vec);
    let prior = Ng::new(0.0, 1.0, 1.0, 1.0, hyper);

    let ftr = Column::new(id, data, prior);
    ColModel::Continuous(ftr)
}

fn gen_all_gauss_state<R: Rng>(
    nrows: usize,
    ncols: usize,
    mut rng: &mut R,
) -> State {
    let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
    for i in 0..ncols {
        ftrs.push(gen_col(i, nrows, &mut rng));
    }
    State::from_prior(
        ftrs,
        Gamma::new(1.0, 1.0).unwrap().into(),
        Gamma::new(1.0, 1.0).unwrap().into(),
        &mut rng,
    )
}

fn load_states<P: AsRef<Path>>(filenames: Vec<P>) -> Vec<State> {
    filenames
        .iter()
        .map(|path| {
            let mut file = File::open(&path).unwrap();
            let mut yaml = String::new();
            let res = file.read_to_string(&mut yaml);
            match res {
                Ok(_) => serde_yaml::from_str(&yaml).unwrap(),
                Err(err) => panic!("Error: {:?}", err),
            }
        })
        .collect()
}

fn get_oracle_from_yaml() -> Oracle {
    let filenames = vec![
        "resources/test/small/small-state-1.yaml",
        "resources/test/small/small-state-2.yaml",
        "resources/test/small/small-state-3.yaml",
    ];
    let states = load_states(filenames);
    let data = DataStore::new(states[0].clone_data());
    Oracle {
        states,
        codebook: Codebook::default(),
        data,
    }
}

fn gen_oracle(nstates: usize) -> Oracle {
    let nrows = 20;
    let ncols = 10;
    let mut rng = rand::thread_rng();
    let states: Vec<State> = (0..nstates)
        .map(|_| gen_all_gauss_state(nrows, ncols, &mut rng))
        .collect();

    let data = DataStore::new(states[0].clone_data());
    Oracle {
        states,
        codebook: Codebook::default(),
        data,
    }
}

#[test]
fn init_from_raw_struct_smoke() {
    let _oracle = gen_oracle(4);
}

#[test]
fn init_from_yaml_files_smoke() {
    let _oracle = get_oracle_from_yaml();
}

#[cfg(test)]
mod depprob {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn values() {
        let oracle = get_oracle_from_yaml();

        assert_relative_eq!(
            oracle.depprob(0, 1).unwrap(),
            1.0 / 3.0,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.depprob(1, 2).unwrap(),
            2.0 / 3.0,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.depprob(0, 2).unwrap(),
            2.0 / 3.0,
            epsilon = 10E-6
        );
    }

    #[test]
    fn bad_first_column_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.depprob(3, 1),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        )
    }

    #[test]
    fn bad_second_column_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.depprob(1, 3),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        )
    }

    #[test]
    fn bad_both_column_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.depprob(4, 3),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        )
    }

    #[test]
    fn bad_first_column_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.depprob_pw(&vec![(3, 1)]),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        )
    }

    #[test]
    fn bad_second_column_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.depprob_pw(&vec![(1, 3)]),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        )
    }

    #[test]
    fn bad_both_column_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.depprob_pw(&vec![(4, 3)]),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        )
    }
}

#[cfg(test)]
mod rowsim {
    use super::*;
    use braid::error::RowSimError;

    #[test]
    fn values() {
        let oracle = get_oracle_from_yaml();

        let rowsim_01 = (0.5 + 0.5 + 0.0) / 3.0;
        let rowsim_12 = (0.5 + 0.5 + 1.0) / 3.0;
        let rowsim_23 = (1.0 + 0.5 + 1.0) / 3.0;

        assert_relative_eq!(
            oracle.rowsim(0, 1, None).unwrap(),
            rowsim_01,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.rowsim(1, 2, None).unwrap(),
            rowsim_12,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.rowsim(2, 3, None).unwrap(),
            rowsim_23,
            epsilon = 10E-6
        );
    }

    #[test]
    fn values_wrt() {
        let oracle = get_oracle_from_yaml();

        let rowsim_01 = (1.0 + 0.0 + 0.0) / 3.0;
        let rowsim_12 = (0.0 + 1.0 + 1.0) / 3.0;
        let rowsim_23 = (1.0 + 0.0 + 1.0) / 3.0;

        let wrt_cols = vec![0];
        let wrt = Some(&wrt_cols);

        assert_relative_eq!(
            oracle.rowsim(0, 1, wrt).unwrap(),
            rowsim_01,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.rowsim(1, 2, wrt).unwrap(),
            rowsim_12,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.rowsim(2, 3, wrt).unwrap(),
            rowsim_23,
            epsilon = 10E-6
        );
    }

    #[test]
    fn duplicate_cols_in_wrt_ignored() {
        let oracle = get_oracle_from_yaml();

        let rowsim_01 = (1.0 + 0.0 + 0.0) / 3.0;
        let rowsim_12 = (0.0 + 1.0 + 1.0) / 3.0;
        let rowsim_23 = (1.0 + 0.0 + 1.0) / 3.0;

        let wrt_cols = vec![0, 0, 0];
        let wrt = Some(&wrt_cols);

        assert_relative_eq!(
            oracle.rowsim(0, 1, wrt).unwrap(),
            rowsim_01,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.rowsim(1, 2, wrt).unwrap(),
            rowsim_12,
            epsilon = 10E-6
        );
        assert_relative_eq!(
            oracle.rowsim(2, 3, wrt).unwrap(),
            rowsim_23,
            epsilon = 10E-6
        );
    }

    #[test]
    fn bad_first_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim(4, 1, None),
            Err(RowSimError::RowIndexOutOfBoundsError)
        );
    }

    #[test]
    fn bad_second_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim(1, 5, None),
            Err(RowSimError::RowIndexOutOfBoundsError)
        );
    }

    #[test]
    fn bad_single_wrt_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim(1, 2, Some(&vec![4])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
        assert_eq!(
            oracle.rowsim(1, 1, Some(&vec![4])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn bad_multi_wrt_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim(1, 2, Some(&vec![0, 5])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
        assert_eq!(
            oracle.rowsim(1, 1, Some(&vec![0, 5])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn empty_vec_in_wrt_causes_error() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim(1, 2, Some(&vec![])),
            Err(RowSimError::EmptyWrtError)
        );
        assert_eq!(
            oracle.rowsim(1, 1, Some(&vec![])),
            Err(RowSimError::EmptyWrtError)
        );
    }

    #[test]
    fn bad_first_row_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim_pw(&vec![(4, 1)], None),
            Err(RowSimError::RowIndexOutOfBoundsError)
        );
    }

    #[test]
    fn bad_second_row_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 5)], None),
            Err(RowSimError::RowIndexOutOfBoundsError)
        );
    }

    #[test]
    fn bad_single_wrt_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 2)], Some(&vec![4])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 1)], Some(&vec![4])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn bad_multi_wrt_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 2)], Some(&vec![0, 5])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 1)], Some(&vec![0, 5])),
            Err(RowSimError::WrtColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn empty_vec_in_wrt_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 2)], Some(&vec![])),
            Err(RowSimError::EmptyWrtError)
        );
        assert_eq!(
            oracle.rowsim_pw(&vec![(1, 1)], Some(&vec![])),
            Err(RowSimError::EmptyWrtError)
        );
    }
}

#[cfg(test)]
mod simulate {
    use super::*;
    use braid::error::{GivenError, SimulateError};
    use braid_stats::Datum;

    #[test]
    fn simulate_single_col_without_given_size_check() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let xs = oracle
            .simulate(&vec![0], &Given::Nothing, 14, None, &mut rng)
            .unwrap();

        assert_eq!(xs.len(), 14);
        assert!(xs.iter().all(|x| x.len() == 1));
    }

    #[test]
    fn simulate_single_col_without_given_single_state_ks() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        // flaky test. try 5 times.
        let ks_pass = (0..5)
            .map(|_| {
                let xs: Vec<f64> = oracle
                    .simulate(
                        &vec![0],
                        &Given::Nothing,
                        1000,
                        Some(vec![0]),
                        &mut rng,
                    )
                    .unwrap()
                    .iter()
                    .map(|row| row[0].to_f64_opt().unwrap())
                    .collect();

                let g1 = Gaussian::new(1.6831137962662617, 4.359431212837638)
                    .unwrap();
                let g2 = Gaussian::new(-0.8244161883997966, 0.7575638719355798)
                    .unwrap();
                let target = Mixture::uniform(vec![g1, g2]).unwrap();

                let (_, ks_p) = rv::misc::ks_test(&xs, |x| target.cdf(&x));

                ks_p
            })
            .any(|ks_p| ks_p > 0.25);

        assert!(ks_pass);
    }

    #[test]
    fn simulate_multi_col_without_given_size_check() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let xs = oracle
            .simulate(&vec![0, 1], &Given::Nothing, 14, None, &mut rng)
            .unwrap();

        assert_eq!(xs.len(), 14);
        assert!(xs.iter().all(|x| x.len() == 2));
    }

    #[test]
    fn no_targets_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result =
            oracle.simulate(&vec![], &Given::Nothing, 14, None, &mut rng);

        assert_eq!(result, Err(SimulateError::NoTargetsError));
    }

    #[test]
    fn oob_targets_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result =
            oracle.simulate(&vec![3], &Given::Nothing, 14, None, &mut rng);

        assert_eq!(result, Err(SimulateError::TargetIndexOutOfBoundsError));
    }

    #[test]
    fn oob_state_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result = oracle.simulate(
            &vec![2],
            &Given::Nothing,
            14,
            Some(vec![3]),
            &mut rng,
        );

        assert_eq!(result, Err(SimulateError::StateIndexOutOfBoundsError));
    }

    #[test]
    fn oob_state_indices_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result = oracle.simulate(
            &vec![2],
            &Given::Nothing,
            14,
            Some(vec![0, 3]),
            &mut rng,
        );

        assert_eq!(result, Err(SimulateError::StateIndexOutOfBoundsError));
    }

    #[test]
    fn no_state_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result = oracle.simulate(
            &vec![2],
            &Given::Nothing,
            14,
            Some(vec![]),
            &mut rng,
        );

        assert_eq!(result, Err(SimulateError::NoStateIndicesError));
    }

    #[test]
    fn same_col_in_target_and_given_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result = oracle.simulate(
            &vec![2],
            &Given::Conditions(vec![(2, Datum::Continuous(1.0))]),
            14,
            None,
            &mut rng,
        );

        assert_eq!(
            result,
            Err(SimulateError::GivenError(
                GivenError::ColumnIndexAppearsInTargetError { col_ix: 2 }
            ))
        );
    }

    #[test]
    fn wrong_datum_type_for_col_in_given_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result = oracle.simulate(
            &vec![1],
            &Given::Conditions(vec![(2, Datum::Categorical(1))]),
            14,
            None,
            &mut rng,
        );

        assert_eq!(
            result,
            Err(SimulateError::GivenError(
                GivenError::InvalidDatumForColumnError { col_ix: 2 }
            ))
        );
    }

    #[test]
    fn oob_condition_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let result = oracle.simulate(
            &vec![1],
            &Given::Conditions(vec![(4, Datum::Categorical(1))]),
            14,
            None,
            &mut rng,
        );

        assert_eq!(
            result,
            Err(SimulateError::GivenError(
                GivenError::ColumnIndexOutOfBoundsError,
            ))
        );
    }

    #[test]
    fn simulate_n_zero_returns_empty_vec() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let xs = oracle
            .simulate(&vec![1], &Given::Nothing, 0, None, &mut rng)
            .unwrap();

        assert!(xs.is_empty());
    }
}

#[cfg(test)]
mod mi {
    use super::*;
    use braid::error::MiError;
    use braid::MiType;

    #[test]
    fn oob_first_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.mi(3, 1, 1_000, MiType::Iqr),
            Err(MiError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_second_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.mi(1, 3, 1_000, MiType::Iqr),
            Err(MiError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn zero_samples_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(oracle.mi(1, 2, 0, MiType::Iqr), Err(MiError::NIsZeroError),);
    }
}

#[cfg(test)]
mod entropy {
    use super::*;
    use braid::error::EntropyError;

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.entropy(&vec![3], 1_000),
            Err(EntropyError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_col_indices_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.entropy(&vec![0, 3], 1_000),
            Err(EntropyError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn no_samples_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.entropy(&vec![0, 1], 0),
            Err(EntropyError::NIsZeroError),
        );
    }

    #[test]
    fn no_targets_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.entropy(&vec![], 1_000),
            Err(EntropyError::NoTargetColumnsError),
        );
    }
}

#[cfg(test)]
mod info_prop {
    use super::*;
    use braid::error::InfoPropError;

    #[test]
    fn oob_target_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.info_prop(&vec![3], &vec![1], 1_000),
            Err(InfoPropError::TargetColumnIndexOutOfBoundsError),
        );

        assert_eq!(
            oracle.info_prop(&vec![0, 3], &vec![1], 1_000),
            Err(InfoPropError::TargetColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_predictor_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.info_prop(&vec![1], &vec![3], 1_000),
            Err(InfoPropError::PredictorColumnIndexOutOfBoundsError),
        );

        assert_eq!(
            oracle.info_prop(&vec![1], &vec![0, 3], 1_000),
            Err(InfoPropError::PredictorColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn no_predictor_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.info_prop(&vec![1], &vec![], 1_000),
            Err(InfoPropError::NoPredictorColumnsError),
        );

        assert_eq!(
            oracle.info_prop(&vec![0, 1], &vec![], 1_000),
            Err(InfoPropError::NoPredictorColumnsError),
        );
    }

    #[test]
    fn no_target_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.info_prop(&vec![], &vec![0], 1_000),
            Err(InfoPropError::NoTargetColumnsError),
        );

        assert_eq!(
            oracle.info_prop(&vec![], &vec![0, 1], 1_000),
            Err(InfoPropError::NoTargetColumnsError),
        );
    }

    #[test]
    fn no_samples_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.info_prop(&vec![1], &vec![0], 0),
            Err(InfoPropError::NIsZeroError),
        );
    }
}

#[cfg(test)]
mod ftype {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.ftype(3),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        );
    }
}

#[cfg(test)]
mod feature_error {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.feature_error(3),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        );
    }
}

#[cfg(test)]
mod summarize {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.summarize_col(3),
            Err(IndexError::ColumnIndexOutOfBoundsError)
        );
    }
}

#[cfg(test)]
mod conditional_entropy {
    use super::*;
    use braid::error::ConditionalEntropyError;
    use braid::ConditionalEntropyType;

    #[test]
    fn oob_target_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let result = oracle.conditional_entropy(3, &vec![2], 1_000);
        assert_eq!(
            result,
            Err(ConditionalEntropyError::TargetColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn oob_predictor_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let result1 = oracle.conditional_entropy(2, &vec![3], 1_000);
        assert_eq!(
            result1,
            Err(ConditionalEntropyError::PredictorColumnIndexOutOfBoundsError)
        );

        let result2 = oracle.conditional_entropy(2, &vec![0, 3], 1_000);
        assert_eq!(
            result2,
            Err(ConditionalEntropyError::PredictorColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn no_predictor_cols_causes_error() {
        let oracle = get_oracle_from_yaml();

        let result = oracle.conditional_entropy(2, &vec![], 1_000);
        assert_eq!(
            result,
            Err(ConditionalEntropyError::NoPredictorColumnsError)
        );
    }

    #[test]
    fn no_samples_causes_error() {
        let oracle = get_oracle_from_yaml();

        let result = oracle.conditional_entropy(2, &vec![1], 0);
        assert_eq!(result, Err(ConditionalEntropyError::NIsZeroError));
    }

    #[test]
    fn duplicate_predictors_cause_error() {
        let oracle = get_oracle_from_yaml();

        let result1 = oracle.conditional_entropy(2, &vec![1, 1], 1_000);
        assert_eq!(
            result1,
            Err(ConditionalEntropyError::DuplicatePredictorsError)
        );

        let result2 = oracle.conditional_entropy(2, &vec![0, 1, 1], 1_000);
        assert_eq!(
            result2,
            Err(ConditionalEntropyError::DuplicatePredictorsError)
        );
    }

    #[test]
    fn oob_target_col_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();

        let result1 = oracle.conditional_entropy_pw(
            &vec![(3, 1)],
            1_000,
            ConditionalEntropyType::UnNormed,
        );
        assert_eq!(
            result1,
            Err(ConditionalEntropyError::TargetColumnIndexOutOfBoundsError)
        );

        let result2 = oracle.conditional_entropy_pw(
            &vec![(3, 1)],
            1_000,
            ConditionalEntropyType::InfoProp,
        );
        assert_eq!(
            result2,
            Err(ConditionalEntropyError::TargetColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn oob_predictor_col_index_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();

        let result1 = oracle.conditional_entropy_pw(
            &vec![(1, 3)],
            1_000,
            ConditionalEntropyType::UnNormed,
        );
        assert_eq!(
            result1,
            Err(ConditionalEntropyError::PredictorColumnIndexOutOfBoundsError)
        );

        let result2 = oracle.conditional_entropy_pw(
            &vec![(1, 3)],
            1_000,
            ConditionalEntropyType::InfoProp,
        );
        assert_eq!(
            result2,
            Err(ConditionalEntropyError::PredictorColumnIndexOutOfBoundsError)
        );
    }

    #[test]
    fn no_pairs_returns_empty_vec() {
        let oracle = get_oracle_from_yaml();

        let result1 = oracle.conditional_entropy_pw(
            &vec![],
            1_000,
            ConditionalEntropyType::UnNormed,
        );
        assert!(result1.unwrap().is_empty());

        let result2 = oracle.conditional_entropy_pw(
            &vec![],
            1_000,
            ConditionalEntropyType::InfoProp,
        );
        assert!(result2.unwrap().is_empty());
    }

    #[test]
    fn no_samples_causes_error_pairwise() {
        let oracle = get_oracle_from_yaml();

        let result1 = oracle.conditional_entropy_pw(
            &vec![(1, 0)],
            0,
            ConditionalEntropyType::UnNormed,
        );
        assert_eq!(result1, Err(ConditionalEntropyError::NIsZeroError));

        let result2 = oracle.conditional_entropy_pw(
            &vec![(1, 0)],
            0,
            ConditionalEntropyType::InfoProp,
        );
        assert_eq!(result2, Err(ConditionalEntropyError::NIsZeroError));
    }
}

#[cfg(test)]
mod surprisal {
    use super::*;
    use braid::error::SurprisalError;
    use braid_stats::Datum;

    #[test]
    fn oob_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.surprisal(&Datum::Continuous(1.0), 4, 1),
            Err(SurprisalError::RowIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.surprisal(&Datum::Continuous(1.0), 1, 3),
            Err(SurprisalError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn wrong_data_type_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.surprisal(&Datum::Categorical(1), 1, 0),
            Err(SurprisalError::InvalidDatumForColumnError),
        );
    }
}

#[cfg(test)]
mod self_surprisal {
    use super::*;
    use braid::error::SurprisalError;

    #[test]
    fn oob_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.self_surprisal(4, 1),
            Err(SurprisalError::RowIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.self_surprisal(1, 3),
            Err(SurprisalError::ColumnIndexOutOfBoundsError),
        );
    }
}

#[cfg(test)]
mod datum {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn oob_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.datum(4, 1),
            Err(IndexError::RowIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.datum(1, 3),
            Err(IndexError::ColumnIndexOutOfBoundsError),
        );
    }
}

#[cfg(test)]
mod draw {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn oob_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        assert_eq!(
            oracle.draw(4, 1, 10, &mut rng),
            Err(IndexError::RowIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        assert_eq!(
            oracle.draw(1, 3, 10, &mut rng),
            Err(IndexError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn no_samples_returns_empty_vec() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        assert!(oracle.draw(1, 2, 0, &mut rng).unwrap().is_empty());
    }
}

#[cfg(test)]
mod impute {
    use super::*;
    use braid::error::IndexError;

    #[test]
    fn oob_row_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.impute(4, 1, None),
            Err(IndexError::RowIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.impute(1, 3, None),
            Err(IndexError::ColumnIndexOutOfBoundsError),
        );
    }
}

#[cfg(test)]
mod predict {
    use super::*;
    use braid::error::{GivenError, PredictError};
    use braid::Given;
    use braid_stats::Datum;

    #[test]
    fn oob_col_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.predict(3, &Given::Nothing, None),
            Err(PredictError::ColumnIndexOutOfBoundsError),
        );
    }

    #[test]
    fn oob_condition_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.predict(
                1,
                &Given::Conditions(vec![(3, Datum::Continuous(1.2))]),
                None
            ),
            Err(PredictError::GivenError(
                GivenError::ColumnIndexOutOfBoundsError
            )),
        );
    }

    #[test]
    fn invalid_condition_datum_type_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.predict(
                1,
                &Given::Conditions(vec![(0, Datum::Categorical(1))]),
                None
            ),
            Err(PredictError::GivenError(
                GivenError::InvalidDatumForColumnError { col_ix: 0 }
            )),
        );
    }

    #[test]
    fn target_in_condition_causes_error() {
        let oracle = get_oracle_from_yaml();

        assert_eq!(
            oracle.predict(
                0,
                &Given::Conditions(vec![(0, Datum::Continuous(2.1))]),
                None
            ),
            Err(PredictError::GivenError(
                GivenError::ColumnIndexAppearsInTargetError { col_ix: 0 }
            )),
        );
    }
}

#[cfg(test)]
mod logp {
    use super::*;
    use braid::error::{GivenError, LogpError};
    use braid::Given;
    use braid_stats::Datum;

    #[test]
    fn oob_target_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 3],
            &vec![vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
            &Given::Nothing,
            None,
        );

        assert_eq!(res, Err(LogpError::TargetIndexOutOfBoundsError));
    }

    #[test]
    fn no_target_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(&vec![], &vec![vec![]], &Given::Nothing, None);

        assert_eq!(res, Err(LogpError::NoTargetsError));
    }

    #[test]
    fn oob_state_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
            &Given::Nothing,
            Some(vec![0, 3]),
        );

        assert_eq!(res, Err(LogpError::StateIndexOutOfBoundsError));
    }

    #[test]
    fn no_state_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
            &Given::Nothing,
            Some(vec![]),
        );

        assert_eq!(res, Err(LogpError::NoStateIndicesError));
    }

    #[test]
    fn too_many_vals_single_row_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0],
            &vec![vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
            &Given::Nothing,
            None,
        );

        assert_eq!(res, Err(LogpError::TargetsIndicesAndValuesMismatchError));
    }

    #[test]
    fn too_many_vals_multi_row_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0],
            &vec![
                vec![Datum::Continuous(4.2)],
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
            ],
            &Given::Nothing,
            None,
        );

        assert_eq!(res, Err(LogpError::TargetsIndicesAndValuesMismatchError));
    }

    #[test]
    fn too_few_vals_single_row_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![vec![Datum::Continuous(2.4)]],
            &Given::Nothing,
            None,
        );

        assert_eq!(res, Err(LogpError::TargetsIndicesAndValuesMismatchError));
    }

    #[test]
    fn too_few_vals_multi_row_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                vec![Datum::Continuous(4.2)],
            ],
            &Given::Nothing,
            None,
        );

        assert_eq!(res, Err(LogpError::TargetsIndicesAndValuesMismatchError));
    }

    #[test]
    fn invalid_datum_type_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                vec![Datum::Continuous(4.3), Datum::Categorical(1)],
            ],
            &Given::Nothing,
            None,
        );

        assert_eq!(
            res,
            Err(LogpError::InvalidDatumForColumnError { col_ix: 1 })
        );
    }

    #[test]
    fn oob_condition_index_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
            ],
            &Given::Conditions(vec![(3, Datum::Continuous(4.0))]),
            None,
        );

        assert_eq!(
            res,
            Err(LogpError::GivenError(
                GivenError::ColumnIndexOutOfBoundsError
            ))
        );
    }

    #[test]
    fn target_in_conditions_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
            ],
            &Given::Conditions(vec![(0, Datum::Continuous(4.0))]),
            None,
        );

        assert_eq!(
            res,
            Err(LogpError::GivenError(
                GivenError::ColumnIndexAppearsInTargetError { col_ix: 0 }
            ))
        );
    }

    #[test]
    fn invalid_datum_type_condition_causes_error() {
        let oracle = get_oracle_from_yaml();

        let res = oracle.logp(
            &vec![0, 1],
            &vec![
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
            ],
            &Given::Conditions(vec![(2, Datum::Categorical(1))]),
            None,
        );

        assert_eq!(
            res,
            Err(LogpError::GivenError(
                GivenError::InvalidDatumForColumnError { col_ix: 2 }
            ))
        );
    }
}

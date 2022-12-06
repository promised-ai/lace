use approx::*;

use std::fs::File;
use std::io::Read;
use std::path::Path;

use braid::cc::feature::{ColModel, Column, FType};
use braid::cc::state::State;
use braid::codebook::Codebook;
use braid::error::IndexError;
use braid::stats::prior::nix::NixHyper;
use braid::{Given, Oracle, OracleT};
use braid_data::{DataStore, SparseContainer};
use rand::Rng;
use rv::dist::{Gamma, Gaussian, Mixture, NormalInvChiSquared};
use rv::traits::{Cdf, Rv};

fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
    let gauss = Gaussian::new(0.0, 1.0).unwrap();
    let hyper = NixHyper::default();
    let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
    let data = SparseContainer::from(data_vec);
    let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);

    let ftr = Column::new(id, data, prior, hyper);
    ColModel::Continuous(ftr)
}

fn gen_all_gauss_state<R: Rng>(
    n_rows: usize,
    n_cols: usize,
    mut rng: &mut R,
) -> State {
    let mut ftrs: Vec<ColModel> = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        ftrs.push(gen_col(i, n_rows, &mut rng));
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
            let mut file = File::open(path).unwrap();
            let mut yaml = String::new();
            let res = file.read_to_string(&mut yaml);
            match res {
                Ok(_) => serde_yaml::from_str(&yaml).unwrap(),
                Err(err) => panic!("Error: {:?}", err),
            }
        })
        .collect()
}

fn dummy_codebook_from_state(state: &State) -> Codebook {
    use braid::cc::feature::Feature;
    use braid::codebook::{ColMetadata, ColType};
    use std::convert::TryInto;
    Codebook {
        table_name: "my_table".into(),
        state_alpha_prior: None,
        view_alpha_prior: None,
        col_metadata: (0..state.n_cols())
            .map(|ix| {
                let ftr = state.feature(ix);
                ColMetadata {
                    name: ix.to_string(),
                    notes: None,
                    coltype: match ftr.ftype() {
                        FType::Continuous => ColType::Continuous {
                            hyper: None,
                            prior: None,
                        },
                        FType::Categorical => ColType::Categorical {
                            k: 2,
                            hyper: None,
                            value_map: None,
                            prior: None,
                        },
                        FType::Count => ColType::Count {
                            hyper: None,
                            prior: None,
                        },
                        _ => panic!("Unsupported coltype"),
                    },
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        row_names: (0..state.n_rows())
            .map(|ix| ix.to_string())
            .collect::<Vec<String>>()
            .try_into()
            .unwrap(),
        comments: None,
    }
}

fn get_oracle_from_yaml() -> Oracle {
    let filenames = vec![
        "resources/test/small/small-state-1.yaml",
        "resources/test/small/small-state-2.yaml",
        "resources/test/small/small-state-3.yaml",
    ];
    let states = load_states(filenames);
    let codebook = dummy_codebook_from_state(&states[0]);
    let data = DataStore::new(states[0].clone_data());
    Oracle {
        states,
        codebook,
        data,
    }
}

fn gen_oracle(n_states: usize) -> Oracle {
    let n_rows = 20;
    let n_cols = 10;
    let mut rng = rand::thread_rng();
    let states: Vec<State> = (0..n_states)
        .map(|_| gen_all_gauss_state(n_rows, n_cols, &mut rng))
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

macro_rules! oracle_test {
    ($oracle_gen: expr) => {
        #[cfg(test)]
        mod depprob {
            use super::*;

            #[test]
            fn values() {
                let oracle = $oracle_gen;

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
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.depprob(3, 1),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3
                    })
                )
            }

            #[test]
            fn bad_second_column_index_causes_error() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.depprob(1, 3),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3
                    })
                )
            }

            #[test]
            fn bad_both_column_index_causes_error() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.depprob(4, 3),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 4
                    })
                )
            }

            #[test]
            fn bad_first_column_index_causes_error_pairwise() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.depprob_pw(&[(3, 1)]),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3
                    })
                )
            }

            #[test]
            fn bad_second_column_index_causes_error_pairwise() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.depprob_pw(&[(1, 3)]),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3
                    })
                )
            }

            #[test]
            fn bad_both_column_index_causes_error_pairwise() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.depprob_pw(&[(4, 3)]),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 4
                    })
                )
            }
        }

        #[cfg(test)]
        mod rowsim {
            use super::*;
            use braid::error::RowSimError;
            use braid::RowSimiliarityVariant;

            #[test]
            fn values() {
                let oracle = $oracle_gen;

                let rowsim_01 = (0.5 + 0.5 + 0.0) / 3.0;
                let rowsim_12 = (0.5 + 0.5 + 1.0) / 3.0;
                let rowsim_23 = (1.0 + 0.5 + 1.0) / 3.0;

                let wrt: Option<&[usize]> = None;

                assert_relative_eq!(
                    oracle
                        .rowsim(
                            0_usize,
                            1_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_01,
                    epsilon = 10E-6
                );
                assert_relative_eq!(
                    oracle
                        .rowsim(
                            1_usize,
                            2_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_12,
                    epsilon = 10E-6
                );
                assert_relative_eq!(
                    oracle
                        .rowsim(
                            2_usize,
                            3_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_23,
                    epsilon = 10E-6
                );
            }

            #[test]
            fn values_col_weighted() {
                let oracle = $oracle_gen;

                let rowsim_01 = (2.0 / 3.0 + 2.0 / 3.0 + 0.0) / 3.0;

                let wrt: Option<&[usize]> = None;

                assert_relative_eq!(
                    oracle
                        .rowsim(
                            0_usize,
                            1_usize,
                            wrt,
                            RowSimiliarityVariant::ColumnWeighted
                        )
                        .unwrap(),
                    rowsim_01,
                    epsilon = 10E-6
                );
            }

            #[test]
            fn values_wrt() {
                let oracle = $oracle_gen;

                let rowsim_01 = (1.0 + 0.0 + 0.0) / 3.0;
                let rowsim_12 = (0.0 + 1.0 + 1.0) / 3.0;
                let rowsim_23 = (1.0 + 0.0 + 1.0) / 3.0;

                let wrt_cols = vec![0];
                let wrt = Some(wrt_cols.as_slice());

                assert_relative_eq!(
                    oracle
                        .rowsim(
                            0_usize,
                            1_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_01,
                    epsilon = 10E-6
                );
                assert_relative_eq!(
                    oracle
                        .rowsim(
                            1_usize,
                            2_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_12,
                    epsilon = 10E-6
                );
                assert_relative_eq!(
                    oracle
                        .rowsim(
                            2_usize,
                            3_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_23,
                    epsilon = 10E-6
                );
            }

            #[test]
            fn duplicate_cols_in_wrt_ignored() {
                let oracle = $oracle_gen;

                let rowsim_01 = (1.0 + 0.0 + 0.0) / 3.0;
                let rowsim_12 = (0.0 + 1.0 + 1.0) / 3.0;
                let rowsim_23 = (1.0 + 0.0 + 1.0) / 3.0;

                let wrt_cols = vec![0, 0, 0];
                let wrt = Some(wrt_cols.as_slice());

                assert_relative_eq!(
                    oracle
                        .rowsim(
                            0_usize,
                            1_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_01,
                    epsilon = 10E-6
                );
                assert_relative_eq!(
                    oracle
                        .rowsim(
                            1_usize,
                            2_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_12,
                    epsilon = 10E-6
                );
                assert_relative_eq!(
                    oracle
                        .rowsim(
                            2_usize,
                            3_usize,
                            wrt,
                            RowSimiliarityVariant::ViewWeighted
                        )
                        .unwrap(),
                    rowsim_23,
                    epsilon = 10E-6
                );
            }

            #[test]
            fn bad_first_row_index_causes_error() {
                let oracle = $oracle_gen;
                let wrt: Option<&[usize]> = None;
                assert_eq!(
                    oracle.rowsim(
                        4_usize,
                        1_usize,
                        wrt,
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::Index(IndexError::RowIndexOutOfBounds {
                        n_rows: 4,
                        row_ix: 4
                    }))
                );
            }

            #[test]
            fn bad_second_row_index_causes_error() {
                let oracle = $oracle_gen;
                let wrt: Option<&[usize]> = None;
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        5_usize,
                        wrt,
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::Index(IndexError::RowIndexOutOfBounds {
                        n_rows: 4,
                        row_ix: 5
                    }))
                );
            }

            #[test]
            fn bad_single_wrt_index_causes_error() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        2_usize,
                        Some(&[4]),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 4
                        }
                    ))
                );
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        1_usize,
                        Some(&[4]),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 4
                        }
                    ))
                );
            }

            #[test]
            fn bad_multi_wrt_index_causes_error() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        2_usize,
                        Some(&[0, 5]),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 5
                        }
                    ))
                );
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        1_usize,
                        Some(&[0, 5]),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 5
                        }
                    ))
                );
            }

            #[test]
            fn empty_vec_in_wrt_causes_error() {
                let oracle = $oracle_gen;
                let empty_wrt: Option<&[usize]> = Some(&[]);
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        2_usize,
                        empty_wrt.clone(),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::EmptyWrt)
                );
                assert_eq!(
                    oracle.rowsim(
                        1_usize,
                        1_usize,
                        empty_wrt,
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::EmptyWrt)
                );
            }

            #[test]
            fn bad_first_row_index_causes_error_pairwise() {
                let oracle = $oracle_gen;

                let wrt: Option<&[usize]> = None;
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(4_usize, 1_usize)],
                        wrt,
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::Index(IndexError::RowIndexOutOfBounds {
                        n_rows: 4,
                        row_ix: 4
                    }))
                );
            }

            #[test]
            fn bad_second_row_index_causes_error_pairwise() {
                let oracle = $oracle_gen;

                let wrt: Option<&[usize]> = None;
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 5_usize)],
                        wrt,
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::Index(IndexError::RowIndexOutOfBounds {
                        n_rows: 4,
                        row_ix: 5
                    }))
                );
            }

            #[test]
            fn bad_single_wrt_index_causes_error_pairwise() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 2_usize)],
                        Some(&[4]),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 4
                        }
                    ))
                );
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 1_usize)],
                        Some(&[4_usize]),
                        RowSimiliarityVariant::ViewWeighted,
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 4
                        }
                    ))
                );
            }

            #[test]
            fn bad_multi_wrt_index_causes_error_pairwise() {
                let oracle = $oracle_gen;
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 2_usize)],
                        Some(&[0_usize, 5_usize]),
                        RowSimiliarityVariant::ViewWeighted,
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 5
                        }
                    ))
                );
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 1_usize)],
                        Some(&[0_usize, 5]),
                        RowSimiliarityVariant::ViewWeighted,
                    ),
                    Err(RowSimError::WrtColumnIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 5
                        }
                    ))
                );
            }

            #[test]
            fn empty_vec_in_wrt_causes_error_pairwise() {
                let oracle = $oracle_gen;

                let wrt: Option<&[usize]> = Some(&[]);
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 2_usize)],
                        wrt.clone(),
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::EmptyWrt)
                );
                assert_eq!(
                    oracle.rowsim_pw(
                        &[(1_usize, 1_usize)],
                        wrt,
                        RowSimiliarityVariant::ViewWeighted
                    ),
                    Err(RowSimError::EmptyWrt)
                );
            }
        }

        #[cfg(test)]
        mod simulate {
            use super::*;
            use braid::error::{GivenError, IndexError, SimulateError};
            use braid_data::Datum;

            #[test]
            fn simulate_single_col_without_given_size_check() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let xs = oracle
                    .simulate(
                        &[0_usize],
                        &Given::<usize>::Nothing,
                        14,
                        None,
                        &mut rng,
                    )
                    .unwrap();

                assert_eq!(xs.len(), 14);
                assert!(xs.iter().all(|x| x.len() == 1));
            }

            #[test]
            fn simulate_single_col_without_given_single_state_ks() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                // flaky test. try 5 times.
                let ks_pass = (0..5)
                    .map(|_| {
                        let xs: Vec<f64> = oracle
                            .simulate(
                                &[0_usize],
                                &Given::<usize>::Nothing,
                                1000,
                                Some(vec![0]),
                                &mut rng,
                            )
                            .unwrap()
                            .iter()
                            .map(|row| row[0].to_f64_opt().unwrap())
                            .collect();

                        let g1 = Gaussian::new(
                            1.6831137962662617,
                            4.359431212837638,
                        )
                        .unwrap();
                        let g2 = Gaussian::new(
                            -0.8244161883997966,
                            0.7575638719355798,
                        )
                        .unwrap();
                        let target = Mixture::uniform(vec![g1, g2]).unwrap();

                        let (_, ks_p) =
                            rv::misc::ks_test(&xs, |x| target.cdf(&x));

                        ks_p
                    })
                    .any(|ks_p| ks_p > 0.25);

                assert!(ks_pass);
            }

            #[test]
            fn simulate_multi_col_without_given_size_check() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let xs = oracle
                    .simulate(
                        &[0, 1],
                        &Given::<usize>::Nothing,
                        14,
                        None,
                        &mut rng,
                    )
                    .unwrap();

                assert_eq!(xs.len(), 14);
                assert!(xs.iter().all(|x| x.len() == 2));
            }

            #[test]
            fn no_targets_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let targets: &[usize] = &[];
                let result = oracle.simulate(
                    targets,
                    &Given::<usize>::Nothing,
                    14,
                    None,
                    &mut rng,
                );

                assert_eq!(result, Err(SimulateError::NoTargets));
            }

            #[test]
            fn oob_targets_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[3_usize],
                    &Given::<usize>::Nothing,
                    14,
                    None,
                    &mut rng,
                );

                assert_eq!(
                    result,
                    Err(SimulateError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            col_ix: 3,
                            n_cols: 3
                        }
                    ))
                );
            }

            #[test]
            fn oob_state_index_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[2_usize],
                    &Given::<usize>::Nothing,
                    14,
                    Some(vec![3]),
                    &mut rng,
                );

                assert_eq!(
                    result,
                    Err(SimulateError::StateIndexOutOfBounds {
                        n_states: 3,
                        state_ix: 3
                    })
                );
            }

            #[test]
            fn oob_state_indices_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[2],
                    &Given::<usize>::Nothing,
                    14,
                    Some(vec![0, 3]),
                    &mut rng,
                );

                assert_eq!(
                    result,
                    Err(SimulateError::StateIndexOutOfBounds {
                        n_states: 3,
                        state_ix: 3
                    })
                );
            }

            #[test]
            fn no_state_index_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[2],
                    &Given::<usize>::Nothing,
                    14,
                    Some(vec![]),
                    &mut rng,
                );

                assert_eq!(result, Err(SimulateError::NoStateIndices));
            }

            #[test]
            fn same_col_in_target_and_given_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[2],
                    &Given::Conditions(vec![(2, Datum::Continuous(1.0))]),
                    14,
                    None,
                    &mut rng,
                );

                assert_eq!(
                    result,
                    Err(SimulateError::GivenError(
                        GivenError::ColumnIndexAppearsInTarget { col_ix: 2 }
                    ))
                );
            }

            #[test]
            fn wrong_datum_type_for_col_in_given_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[1],
                    &Given::Conditions(vec![(2, Datum::Categorical(1))]),
                    14,
                    None,
                    &mut rng,
                );

                assert_eq!(
                    result,
                    Err(SimulateError::GivenError(
                        GivenError::InvalidDatumForColumn {
                            col_ix: 2,
                            ftype_req: FType::Categorical,
                            ftype: FType::Continuous
                        }
                    ))
                );
            }

            #[test]
            fn oob_condition_index_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let result = oracle.simulate(
                    &[1],
                    &Given::Conditions(vec![(4, Datum::Categorical(1))]),
                    14,
                    None,
                    &mut rng,
                );

                assert_eq!(
                    result,
                    Err(SimulateError::GivenError(GivenError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 4
                        }
                    )))
                );
            }

            #[test]
            fn simulate_n_zero_returns_empty_vec() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                let xs = oracle
                    .simulate(&[1], &Given::<usize>::Nothing, 0, None, &mut rng)
                    .unwrap();

                assert!(xs.is_empty());
            }
        }

        #[cfg(test)]
        mod mi {
            use super::*;
            use braid::error::{IndexError, MiError};
            use braid::MiType;

            #[test]
            fn oob_first_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.mi(3, 1, 1_000, MiType::Iqr),
                    Err(MiError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn oob_second_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.mi(1, 3, 1_000, MiType::Iqr),
                    Err(MiError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn zero_samples_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.mi(1, 2, 0, MiType::Iqr),
                    Err(MiError::NIsZero),
                );
            }
        }

        #[cfg(test)]
        mod entropy {
            use super::*;
            use braid::error::{EntropyError, IndexError};

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.entropy(&[3], 1_000),
                    Err(EntropyError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn oob_col_indices_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.entropy(&[0, 3], 1_000),
                    Err(EntropyError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn no_samples_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.entropy(&[0, 1], 0),
                    Err(EntropyError::NIsZero),
                );
            }

            #[test]
            fn no_targets_causes_error() {
                let oracle = $oracle_gen;

                let targets: &[usize] = &[];
                assert_eq!(
                    oracle.entropy(targets, 1_000),
                    Err(EntropyError::NoTargetColumns),
                );
            }
        }

        #[cfg(test)]
        mod info_prop {
            use super::*;
            use braid::error::InfoPropError;

            #[test]
            fn oob_target_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.info_prop(&[3], &[1], 1_000),
                    Err(InfoPropError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );

                assert_eq!(
                    oracle.info_prop(&[0, 3], &[1], 1_000),
                    Err(InfoPropError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn oob_predictor_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.info_prop(&[1], &[3], 1_000),
                    Err(InfoPropError::PredictorIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3
                        }
                    )),
                );

                assert_eq!(
                    oracle.info_prop(&[1], &[0, 3], 1_000),
                    Err(InfoPropError::PredictorIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn no_predictor_index_causes_error() {
                let oracle = $oracle_gen;

                let empty: &[usize] = &[];
                assert_eq!(
                    oracle.info_prop(&[1], empty, 1_000),
                    Err(InfoPropError::NoPredictorColumns),
                );

                assert_eq!(
                    oracle.info_prop(&[0, 1], empty, 1_000),
                    Err(InfoPropError::NoPredictorColumns),
                );
            }

            #[test]
            fn no_target_index_causes_error() {
                let oracle = $oracle_gen;

                let empty: &[usize] = &[];
                assert_eq!(
                    oracle.info_prop(empty, &[0], 1_000),
                    Err(InfoPropError::NoTargetColumns),
                );

                assert_eq!(
                    oracle.info_prop(empty, &[0, 1], 1_000),
                    Err(InfoPropError::NoTargetColumns),
                );
            }

            #[test]
            fn no_samples_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.info_prop(&[1], &[0], 0),
                    Err(InfoPropError::NIsZero),
                );
            }
        }

        #[cfg(test)]
        mod ftype {
            use super::*;
            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.ftype(3),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3,
                    })
                );
            }
        }

        #[cfg(test)]
        mod feature_error {
            use super::*;
            use braid::error::IndexError;

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.feature_error(3),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3,
                    })
                );
            }
        }

        #[cfg(test)]
        mod summarize {
            use super::*;
            use braid::error::IndexError;

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.summarize_col(3),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3,
                    })
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
                let oracle = $oracle_gen;

                let result = oracle.conditional_entropy(3, &[2], 1_000);
                assert_eq!(
                    result,
                    Err(ConditionalEntropyError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    ))
                );
            }

            #[test]
            fn oob_predictor_col_index_causes_error() {
                let oracle = $oracle_gen;

                let result1 = oracle.conditional_entropy(2, &[3], 1_000);
                assert_eq!(
                    result1,
                    Err(ConditionalEntropyError::PredictorIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    ))
                );

                let result2 = oracle.conditional_entropy(2, &[0, 3], 1_000);
                assert_eq!(
                    result2,
                    Err(ConditionalEntropyError::PredictorIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    ))
                );
            }

            #[test]
            fn no_predictor_cols_causes_error() {
                let oracle = $oracle_gen;

                let empty: &[usize] = &[];
                let result = oracle.conditional_entropy(2, empty, 1_000);
                assert_eq!(
                    result,
                    Err(ConditionalEntropyError::NoPredictorColumns)
                );
            }

            #[test]
            fn no_samples_causes_error() {
                let oracle = $oracle_gen;

                let result = oracle.conditional_entropy(2, &[1], 0);
                assert_eq!(result, Err(ConditionalEntropyError::NIsZero));
            }

            #[test]
            fn duplicate_predictors_cause_error() {
                let oracle = $oracle_gen;

                let result1 = oracle.conditional_entropy(2, &[1, 1], 1_000);
                assert_eq!(
                    result1,
                    Err(ConditionalEntropyError::DuplicatePredictors {
                        col_ix: 1
                    })
                );

                let result2 = oracle.conditional_entropy(2, &[0, 1, 1], 1_000);
                assert_eq!(
                    result2,
                    Err(ConditionalEntropyError::DuplicatePredictors {
                        col_ix: 1
                    })
                );
            }

            #[test]
            fn oob_target_col_index_causes_error_pairwise() {
                let oracle = $oracle_gen;

                let result1 = oracle.conditional_entropy_pw(
                    &[(3, 1)],
                    1_000,
                    ConditionalEntropyType::UnNormed,
                );
                assert_eq!(
                    result1,
                    Err(ConditionalEntropyError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            col_ix: 3,
                            n_cols: 3
                        }
                    ))
                );

                let result2 = oracle.conditional_entropy_pw(
                    &[(3, 1)],
                    1_000,
                    ConditionalEntropyType::InfoProp,
                );
                assert_eq!(
                    result2,
                    Err(ConditionalEntropyError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3
                        }
                    ))
                );
            }

            #[test]
            fn oob_predictor_col_index_causes_error_pairwise() {
                let oracle = $oracle_gen;

                let result1 = oracle.conditional_entropy_pw(
                    &[(1, 3)],
                    1_000,
                    ConditionalEntropyType::UnNormed,
                );
                assert_eq!(
                    result1,
                    Err(ConditionalEntropyError::PredictorIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            col_ix: 3,
                            n_cols: 3
                        }
                    ))
                );

                let result2 = oracle.conditional_entropy_pw(
                    &[(1, 3)],
                    1_000,
                    ConditionalEntropyType::InfoProp,
                );
                assert_eq!(
                    result2,
                    Err(ConditionalEntropyError::PredictorIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            col_ix: 3,
                            n_cols: 3
                        }
                    ))
                );
            }

            #[test]
            fn no_pairs_returns_empty_vec() {
                let oracle = $oracle_gen;

                let empty: &[(usize, usize)] = &[];
                let result1 = oracle.conditional_entropy_pw(
                    empty,
                    1_000,
                    ConditionalEntropyType::UnNormed,
                );
                assert!(result1.unwrap().is_empty());

                let result2 = oracle.conditional_entropy_pw(
                    empty,
                    1_000,
                    ConditionalEntropyType::InfoProp,
                );
                assert!(result2.unwrap().is_empty());
            }

            #[test]
            fn no_samples_causes_error_pairwise() {
                let oracle = $oracle_gen;

                let result1 = oracle.conditional_entropy_pw(
                    &[(1, 0)],
                    0,
                    ConditionalEntropyType::UnNormed,
                );
                assert_eq!(result1, Err(ConditionalEntropyError::NIsZero));

                let result2 = oracle.conditional_entropy_pw(
                    &[(1, 0)],
                    0,
                    ConditionalEntropyType::InfoProp,
                );
                assert_eq!(result2, Err(ConditionalEntropyError::NIsZero));
            }
        }

        #[cfg(test)]
        mod surprisal {
            use super::*;
            use braid::error::{IndexError, SurprisalError};
            use braid_data::Datum;

            #[test]
            fn oob_row_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.surprisal(&Datum::Continuous(1.0), 4, 1, None),
                    Err(SurprisalError::IndexError(
                        IndexError::RowIndexOutOfBounds {
                            row_ix: 4,
                            n_rows: 4,
                        }
                    )),
                );
            }

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.surprisal(&Datum::Continuous(1.0), 1, 3, None),
                    Err(SurprisalError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn wrong_data_type_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.surprisal(&Datum::Categorical(1), 1, 0, None),
                    Err(SurprisalError::InvalidDatumForColumn {
                        col_ix: 0,
                        ftype_req: FType::Categorical,
                        ftype: FType::Continuous,
                    })
                );
            }
        }

        #[cfg(test)]
        mod self_surprisal {
            use super::*;
            use braid::error::{IndexError, SurprisalError};

            #[test]
            fn oob_row_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.self_surprisal(4, 1, None),
                    Err(SurprisalError::IndexError(
                        IndexError::RowIndexOutOfBounds {
                            n_rows: 4,
                            row_ix: 4
                        }
                    )),
                );
            }

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.self_surprisal(1, 3, None),
                    Err(SurprisalError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }
        }

        #[cfg(test)]
        mod datum {
            use super::*;
            use braid::error::IndexError;

            #[test]
            fn oob_row_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.datum(4, 1),
                    Err(IndexError::RowIndexOutOfBounds {
                        n_rows: 4,
                        row_ix: 4
                    }),
                );
            }

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.datum(1, 3),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3
                    }),
                );
            }
        }

        #[cfg(test)]
        mod draw {
            use super::*;
            use braid::error::IndexError;

            #[test]
            fn oob_row_index_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                assert_eq!(
                    oracle.draw(4, 1, 10, &mut rng),
                    Err(IndexError::RowIndexOutOfBounds {
                        row_ix: 4,
                        n_rows: 4
                    }),
                );
            }

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;
                let mut rng = rand::thread_rng();

                assert_eq!(
                    oracle.draw(1, 3, 10, &mut rng),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3,
                    }),
                );
            }

            #[test]
            fn no_samples_returns_empty_vec() {
                let oracle = $oracle_gen;
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
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.impute(4, 1, None),
                    Err(IndexError::RowIndexOutOfBounds {
                        n_rows: 4,
                        row_ix: 4,
                    }),
                );
            }

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.impute(1, 3, None),
                    Err(IndexError::ColumnIndexOutOfBounds {
                        n_cols: 3,
                        col_ix: 3,
                    }),
                );
            }
        }

        #[cfg(test)]
        mod predict {
            use super::*;
            use braid::error::{GivenError, IndexError, PredictError};
            use braid::Given;
            use braid_data::Datum;

            #[test]
            fn oob_col_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.predict(3, &Given::<usize>::Nothing, None, None),
                    Err(PredictError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )),
                );
            }

            #[test]
            fn oob_condition_index_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.predict(
                        1,
                        &Given::Conditions(vec![(3, Datum::Continuous(1.2))]),
                        None,
                        None
                    ),
                    Err(PredictError::GivenError(GivenError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    ))),
                );
            }

            #[test]
            fn invalid_condition_datum_type_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.predict(
                        1,
                        &Given::Conditions(vec![(0, Datum::Categorical(1))]),
                        None,
                        None,
                    ),
                    Err(PredictError::GivenError(
                        GivenError::InvalidDatumForColumn {
                            col_ix: 0,
                            ftype_req: FType::Categorical,
                            ftype: FType::Continuous,
                        }
                    )),
                );
            }

            #[test]
            fn target_in_condition_causes_error() {
                let oracle = $oracle_gen;

                assert_eq!(
                    oracle.predict(
                        0,
                        &Given::Conditions(vec![(0, Datum::Continuous(2.1))]),
                        None,
                        None
                    ),
                    Err(PredictError::GivenError(
                        GivenError::ColumnIndexAppearsInTarget { col_ix: 0 }
                    )),
                );
            }
        }

        #[cfg(test)]
        mod logp {
            use super::*;
            use braid::error::{GivenError, IndexError, LogpError};
            use braid::Given;
            use braid_data::Datum;

            #[test]
            fn oob_target_index_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 3],
                    &[vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::TargetIndexOutOfBounds(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    ))
                );
            }

            #[test]
            fn no_target_index_causes_error() {
                let oracle = $oracle_gen;

                let col_ixs: &[usize] = &[];
                let res = oracle.logp(
                    col_ixs,
                    &[vec![]],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(res, Err(LogpError::NoTargets));
            }

            #[test]
            fn oob_state_index_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
                    &Given::<usize>::Nothing,
                    Some(&[0, 3]),
                );

                assert_eq!(
                    res,
                    Err(LogpError::StateIndexOutOfBounds {
                        n_states: 3,
                        state_ix: 3,
                    })
                );
            }

            #[test]
            fn no_state_index_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
                    &Given::<usize>::Nothing,
                    Some(&[]),
                );

                assert_eq!(res, Err(LogpError::NoStateIndices));
            }

            #[test]
            fn too_many_vals_single_row_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0],
                    &[vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::TargetsIndicesAndValuesMismatch {
                        nvals: 2,
                        ntargets: 1,
                    })
                );
            }

            #[test]
            fn too_many_vals_multi_row_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0],
                    &[
                        vec![Datum::Continuous(4.2)],
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                    ],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::TargetsIndicesAndValuesMismatch {
                        nvals: 2,
                        ntargets: 1,
                    })
                );
            }

            #[test]
            fn too_few_vals_single_row_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[vec![Datum::Continuous(2.4)]],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::TargetsIndicesAndValuesMismatch {
                        nvals: 1,
                        ntargets: 2,
                    })
                );
            }

            #[test]
            fn too_few_vals_multi_row_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                        vec![Datum::Continuous(4.2)],
                    ],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::TargetsIndicesAndValuesMismatch {
                        ntargets: 2,
                        nvals: 1,
                    })
                );
            }

            #[test]
            fn invalid_datum_type_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                        vec![Datum::Continuous(4.3), Datum::Categorical(1)],
                    ],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::InvalidDatumForColumn {
                        col_ix: 1,
                        ftype_req: FType::Categorical,
                        ftype: FType::Continuous,
                    })
                );
            }

            #[test]
            fn oob_condition_index_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                    ],
                    &Given::Conditions(vec![(3, Datum::Continuous(4.0))]),
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::GivenError(GivenError::IndexError(
                        IndexError::ColumnIndexOutOfBounds {
                            n_cols: 3,
                            col_ix: 3,
                        }
                    )))
                );
            }

            #[test]
            fn target_in_conditions_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                    ],
                    &Given::Conditions(vec![(0, Datum::Continuous(4.0))]),
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::GivenError(
                        GivenError::ColumnIndexAppearsInTarget { col_ix: 0 }
                    ))
                );
            }

            #[test]
            fn invalid_datum_type_condition_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                        vec![Datum::Continuous(1.2), Datum::Continuous(2.4)],
                    ],
                    &Given::Conditions(vec![(2, Datum::Categorical(1))]),
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::GivenError(
                        GivenError::InvalidDatumForColumn {
                            col_ix: 2,
                            ftype_req: FType::Categorical,
                            ftype: FType::Continuous,
                        }
                    ))
                );
            }

            #[test]
            fn missing_value_in_target_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 2],
                    &[vec![Datum::Continuous(1.2), Datum::Missing]],
                    &Given::<usize>::Nothing,
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::RequestedLogpOfMissing { col_ix: 2 })
                );
            }

            #[test]
            fn missing_datum_in_condition_causes_error() {
                let oracle = $oracle_gen;

                let res = oracle.logp(
                    &[0, 1],
                    &[vec![Datum::Continuous(1.2), Datum::Continuous(2.4)]],
                    &Given::Conditions(vec![(2, Datum::Missing)]),
                    None,
                );

                assert_eq!(
                    res,
                    Err(LogpError::GivenError(GivenError::MissingDatum {
                        col_ix: 2,
                    }))
                );
            }
        }
    };
}

#[cfg(test)]
mod oracle {
    use super::*;
    oracle_test!(get_oracle_from_yaml());
}

#[cfg(test)]
mod dataless {
    use super::*;
    use braid::DatalessOracle;
    oracle_test!(DatalessOracle::from(get_oracle_from_yaml()));
}

#[cfg(test)]
mod engine {
    use super::*;
    use braid::Engine;
    oracle_test!(Engine::from(get_oracle_from_yaml()));
}

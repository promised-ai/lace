extern crate rand;
extern crate serde_yaml;

use std::cmp::PartialOrd;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use self::rand::Rng;

use cc::{ColModel, DType, State};
use dist::traits::{Distribution, KlDivergence};
use dist::{Categorical, Gaussian, MixtureModel};
use interface::Given;
use misc::{argmax, logsumexp};
use optimize::fmin_bounded;
use special::digamma;

// Helper functions
// ----------------
pub fn load_states(filenames: Vec<&str>) -> Vec<State> {
    filenames
        .iter()
        .map(|filename| {
            let path = Path::new(&filename);
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

// Weight Calculation
// ------------------
pub fn given_weights(
    states: &Vec<State>,
    col_ixs: &Vec<usize>,
    given_opt: &Option<Vec<(usize, DType)>>,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    let mut state_weights: Vec<_> = Vec::with_capacity(states.len());

    for state in states {
        let view_weights =
            single_state_weights(&state, &col_ixs, &given_opt, false);
        state_weights.push(view_weights);
    }
    state_weights
}

pub fn single_state_weights(
    state: &State,
    col_ixs: &Vec<usize>,
    given_opt: &Option<Vec<(usize, DType)>>,
    weightless: bool,
) -> BTreeMap<usize, Vec<f64>> {
    let mut view_ixs: HashSet<usize> = HashSet::new();
    col_ixs.iter().for_each(|col_ix| {
        let view_ix = state.asgn.asgn[*col_ix];
        view_ixs.insert(view_ix);
    });

    let mut view_weights: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    view_ixs.iter().for_each(|&view_ix| {
        let weights =
            single_view_weights(&state, view_ix, &given_opt, weightless);
        view_weights.insert(view_ix, weights);
    });
    view_weights
}

pub fn single_view_weights(
    state: &State,
    target_view_ix: usize,
    given_opt: &Option<Vec<(usize, DType)>>,
    weightless: bool,
) -> Vec<f64> {
    let view = &state.views[target_view_ix];

    let mut weights = if weightless {
        vec![0.0; view.asgn.ncats]
    } else {
        view.asgn.log_weights()
    };

    match given_opt {
        &Some(ref given) => for &(id, ref datum) in given {
            let in_target_view = state.asgn.asgn[id] == target_view_ix;
            if in_target_view {
                weights = view.ftrs[&id].accum_weights(&datum, weights);
            }
        },
        &None => (),
    }
    weights
}

// Probability calculation
// -----------------------
pub fn state_logp(
    state: &State,
    col_ixs: &Vec<usize>,
    vals: &Vec<Vec<DType>>,
    given_opt: &Option<Vec<(usize, DType)>>,
) -> Vec<f64> {
    let mut view_weights =
        single_state_weights(state, &col_ixs, &given_opt, false);

    // normalize view weights
    for weights in view_weights.values_mut() {
        let logz = logsumexp(weights);
        weights.iter_mut().for_each(|w| *w -= logz);
    }

    vals.iter()
        .map(|val| single_val_logp(&state, &col_ixs, &val, &view_weights))
        .collect()
}

pub fn single_val_logp(
    state: &State,
    col_ixs: &Vec<usize>,
    val: &Vec<DType>,
    view_weights: &BTreeMap<usize, Vec<f64>>,
) -> f64 {
    // turn col_ixs and values into a given
    let mut obs = Vec::with_capacity(col_ixs.len());
    for (&col_ix, datum) in col_ixs.iter().zip(val) {
        obs.push((col_ix, datum.clone()));
    }

    // compute the un-normalied, 'weightless', weights using the given
    let logp_obs = single_state_weights(state, &col_ixs, &Some(obs), true);

    // add everything up
    let mut logp_out = 0.0;
    for (view_ix, mut logps) in logp_obs {
        let weights = &view_weights[&view_ix];
        logps
            .iter_mut()
            .zip(weights.iter())
            .for_each(|(lp, w)| *lp += w);
        logp_out += logsumexp(&logps);
    }
    logp_out
}

// mutual information
// ------------------
/// K-nearest neighbor mutual information approximation
pub fn knn_mi(xy: &Vec<(f64, f64)>, k_opt: Option<usize>) -> f64 {
    let xs: Vec<f64> = xy.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = xy.iter().map(|(_, y)| *y).collect();
    knn_mi_uz(&xs, &ys, k_opt)
}

/// K-nearest neighbor mutual information approximation on unzipped data
pub fn knn_mi_uz(xs: &Vec<f64>, ys: &Vec<f64>, k_opt: Option<usize>) -> f64 {
    let k: usize = k_opt.unwrap_or(3);
    let n = xs.len();
    let logn = (n as f64).ln();
    let digamma_k = digamma(k as f64);

    xs.iter()
        .zip(ys.iter())
        .enumerate()
        .fold(0.0, |acc, (i, (xi, yi))| {
            let dx: Vec<f64> = knn_mi_dist(*xi, &xs, i);
            let dy: Vec<f64> = knn_mi_dist(*yi, &ys, i);
            let mut ds: Vec<f64> = dx.iter()
                .zip(dy.iter())
                .map(|(&dxi, &dyi)| dxi.max(dyi))
                .collect();

            ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let rho = ds[k - 1];

            let digmma_ki = if rho == 0.0 {
                let ki: f64 =
                    ds.iter().fold(
                        0.0,
                        |acc, &d| if d == 0.0 { acc + d } else { acc },
                    );
                digamma(ki)
            } else {
                digamma_k
            };

            let nx = knn_mi_d_count(rho, &dx);
            let ny = knn_mi_d_count(rho, &dy);

            acc + digmma_ki + logn - (ny + 1.0).ln() - (nx + 1.0).ln()
        }) / (n as f64)
}

fn knn_mi_d_count(rho: f64, dx: &Vec<f64>) -> f64 {
    dx.iter().fold(
        0.0,
        |acc, &dxi| {
            if dxi <= rho {
                acc + 1.0
            } else {
                acc
            }
        },
    )
}

/// Compute the L2 distance for between xi and each eleement in xs except for
/// xi.
fn knn_mi_dist(xi: f64, xs: &Vec<f64>, i: usize) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .filter_map(|(j, xj)| {
            if j == i {
                None
            } else {
                Some((xi - xj).powi(2))
            }
        })
        .collect()
}

// Imputation
// ----------
fn impute_bounds(states: &Vec<State>, col_ix: usize) -> (f64, f64) {
    let (lowers, uppers): (Vec<f64>, Vec<f64>) = states
        .iter()
        .map(|state| state.impute_bounds(col_ix).unwrap())
        .unzip();
    let min: f64 = lowers.iter().cloned().fold(0.0 / 0.0, f64::min);
    let max: f64 = uppers.iter().cloned().fold(0.0 / 0.0, f64::max);
    assert!(min <= max);
    (min, max)
}

pub fn continuous_impute(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> f64 {
    let cpnts: Vec<&Gaussian> = states
        .iter()
        .map(|state| {
            state
                .extract_continuous_cpnt(row_ix, col_ix)
                .unwrap()
        })
        .collect();

    if cpnts.len() == 1 {
        cpnts[0].mu
    } else {
        let f = |x: f64| {
            let logfs: Vec<f64> = cpnts
                .iter()
                .map(|&cpnt| cpnt.loglike(&x))
                .collect();
            -logsumexp(&logfs)
        };

        let bounds = impute_bounds(&states, col_ix);
        fmin_bounded(f, bounds, None, None)
    }
}

pub fn categorical_impute(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> u8 {
    let cpnts: Vec<&Categorical<u8>> = states
        .iter()
        .map(|state| {
            state
                .extract_categorical_cpnt(row_ix, col_ix)
                .unwrap()
        })
        .collect();

    let k = cpnts[0].log_weights.len() as u8;
    let fs: Vec<f64> = (0..k)
        .map(|x| {
            let logfs: Vec<f64> = cpnts
                .iter()
                .map(|&cpnt| cpnt.loglike(&x))
                .collect();
            logsumexp(&logfs)
        })
        .collect();
    argmax(&fs) as u8
}

// Prediction
// ----------
pub fn continuous_predict(
    states: &Vec<State>,
    col_ix: usize,
    given: &Given,
) -> f64 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let f = |x: f64| {
        let y: Vec<Vec<DType>> = vec![vec![DType::Continuous(x)]];
        let scores: Vec<f64> = states
            .iter()
            .map(|state| state_logp(state, &col_ixs, &y, &given)[0])
            .collect();
        -logsumexp(&scores)
    };

    let bounds = impute_bounds(&states, col_ix);
    fmin_bounded(f, bounds, None, None)
}

pub fn categorical_predict(
    states: &Vec<State>,
    col_ix: usize,
    given: &Given,
) -> u8 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let f = |x: u8| {
        let y: Vec<Vec<DType>> = vec![vec![DType::Categorical(x)]];
        let scores: Vec<f64> = states
            .iter()
            .map(|state| state_logp(state, &col_ixs, &y, &given)[0])
            .collect();
        logsumexp(&scores)
    };

    let k: u8 = match states[0].get_feature(col_ix) {
        ColModel::Categorical(ftr) => ftr.prior.dir.k as u8,
        _ => panic!("FType mitmatch."),
    };

    let fs: Vec<f64> = (0..k).map(|x| f(x)).collect();
    argmax(&fs) as u8
}

// Predictive uncertainty helpers
// ------------------------------
// FIXME: this code also makes me want to kill myself
pub fn js_uncertainty(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
    n_samples: usize,
    mut rng: &mut Rng,
) -> f64 {
    let nstates = states.len();
    let view_ix = states[0].asgn.asgn[col_ix];
    let view = &states[0].views[view_ix];
    let k = view.asgn.asgn[row_ix];
    match &view.ftrs[&col_ix] {
        &ColModel::Continuous(ref ftr) => {
            let mut cpnts = Vec::with_capacity(nstates);
            cpnts.push(ftr.components[k].clone());
            for i in 1..nstates {
                let view_ix_s = states[i].asgn.asgn[col_ix];
                let view_s = &states[i].views[view_ix_s];
                let k_s = view.asgn.asgn[row_ix];
                match &view_s.ftrs[&col_ix] {
                    &ColModel::Continuous(ref ftr) => {
                        cpnts.push(ftr.components[k_s].clone());
                    }
                    _ => panic!("Mismatched feature type"),
                }
            }
            let m = MixtureModel::flat(cpnts);
            m.js_divergence(n_samples, &mut rng)
        }
        &ColModel::Categorical(ref ftr) => {
            let mut cpnts = Vec::with_capacity(nstates);
            cpnts.push(ftr.components[k].clone());
            for i in 1..nstates {
                let view_ix_s = states[i].asgn.asgn[col_ix];
                let view_s = &states[i].views[view_ix_s];
                let k_s = view.asgn.asgn[row_ix];
                match &view_s.ftrs[&col_ix] {
                    &ColModel::Categorical(ref ftr) => {
                        cpnts.push(ftr.components[k_s].clone());
                    }
                    _ => panic!("Mismatched feature type"),
                }
            }
            let m = MixtureModel::flat(cpnts);
            m.js_divergence(n_samples, &mut rng)
        }
    }
}

pub fn kl_uncertainty(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> f64 {
    let locators: Vec<(usize, usize)> = states
        .iter()
        .map(|state| {
            let view_ix = state.asgn.asgn[col_ix];
            let cpnt_ix = state.views[view_ix].asgn.asgn[row_ix];
            (view_ix, cpnt_ix)
        })
        .collect();

    // FIXME: this code makes me want to die
    let mut kl_sum = 0.0;
    for (i, &(vi, ki)) in locators.iter().enumerate() {
        let cm_i = &states[i].views[vi].ftrs[&col_ix];
        match cm_i {
            &ColModel::Continuous(ref fi) => {
                let cpnt_i = &fi.components[ki];
                for (j, &(vj, kj)) in locators.iter().enumerate() {
                    if i != j {
                        let cm_j = &states[j].views[vj].ftrs[&col_ix];
                        match cm_j {
                            &ColModel::Continuous(ref fj) => {
                                let cpnt_j = &fj.components[kj];
                                kl_sum += cpnt_i.kl_divergence(cpnt_j);
                            }
                            _ => panic!("2nd ColModel was not continuous"),
                        }
                    }
                }
            }
            &ColModel::Categorical(ref fi) => {
                let cpnt_i = &fi.components[ki];
                for (j, &(vj, kj)) in locators.iter().enumerate() {
                    if i != j {
                        let cm_j = &states[j].views[vj].ftrs[&col_ix];
                        match cm_j {
                            &ColModel::Categorical(ref fj) => {
                                let cpnt_j = &fj.components[kj];
                                kl_sum += cpnt_i.kl_divergence(cpnt_j);
                            }
                            _ => panic!("2nd ColModel was not categorical"),
                        }
                    }
                }
            }
        }
    }

    let nstates = states.len() as f64;
    kl_sum / (nstates * nstates - nstates)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    fn get_single_continuous_state_from_yaml() -> State {
        let filenames = vec!["resources/test/single-continuous.yaml"];
        load_states(filenames).remove(0)
    }

    fn get_single_categorical_state_from_yaml() -> State {
        let filenames = vec!["resources/test/single-categorical.yaml"];
        load_states(filenames).remove(0)
    }

    fn get_states_from_yaml() -> Vec<State> {
        let filenames = vec![
            "resources/test/small-state-1.yaml",
            "resources/test/small-state-2.yaml",
            "resources/test/small-state-3.yaml",
        ];
        load_states(filenames)
    }

    #[test]
    fn single_continuous_column_weights_no_given() {
        let state = get_single_continuous_state_from_yaml();

        let weights = single_view_weights(&state, 0, &None, false);

        assert_relative_eq!(weights[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights[1], -0.6931471805599453, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given() {
        let state = get_single_continuous_state_from_yaml();
        let given = Some(vec![(0, DType::Continuous(0.5))]);

        let weights = single_view_weights(&state, 0, &given, false);

        assert_relative_eq!(weights[0], -2.8570549170130315, epsilon = TOL);
        assert_relative_eq!(weights[1], -16.59893853320467, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given_weightless() {
        let state = get_single_continuous_state_from_yaml();
        let given = Some(vec![(0, DType::Continuous(0.5))]);

        let weights = single_view_weights(&state, 0, &given, true);

        assert_relative_eq!(weights[0], -2.1639077364530861, epsilon = TOL);
        assert_relative_eq!(weights[1], -15.905791352644725, epsilon = TOL);
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let states = get_states_from_yaml();

        let weights_0 = single_view_weights(&states[0], 0, &None, false);

        assert_relative_eq!(weights_0[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -0.6931471805599453, epsilon = TOL);

        let weights_1 = single_view_weights(&states[0], 1, &None, false);

        assert_relative_eq!(weights_1[0], -1.3862943611198906, epsilon = TOL);
        assert_relative_eq!(weights_1[1], -0.2876820724517809, epsilon = TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_one_given() {
        let states = get_states_from_yaml();

        // column 1 should not affect view 0 weights because it is assigned to
        // view 1
        let given = Some(vec![
            (0, DType::Continuous(0.0)),
            (1, DType::Continuous(-1.0)),
        ]);

        let weights_0 = single_view_weights(&states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -3.1589583681201292, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -1.9265784475169849, epsilon = TOL);

        let weights_1 = single_view_weights(&states[0], 1, &given, false);

        assert_relative_eq!(weights_1[0], -4.0958633027669231, epsilon = TOL);
        assert_relative_eq!(weights_1[1], -0.4177811369331429, epsilon = TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_added_given() {
        let states = get_states_from_yaml();

        let given = Some(vec![
            (0, DType::Continuous(0.0)),
            (2, DType::Continuous(-1.0)),
        ]);

        let weights_0 = single_view_weights(&states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -5.6691757676902537, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -9.3045547861934459, epsilon = TOL);
    }

    #[test]
    fn single_state_weights_value_check() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let given = Some(vec![
            (0, DType::Continuous(0.0)),
            (1, DType::Continuous(-1.0)),
            (2, DType::Continuous(-1.0)),
        ]);

        let weights = single_state_weights(&states[0], &col_ixs, &given, false);

        assert_eq!(weights.len(), 2);
        assert_eq!(weights[&0].len(), 2);
        assert_eq!(weights[&1].len(), 2);

        assert_relative_eq!(
            weights[&0][0],
            -5.6691757676902537,
            epsilon = TOL
        );
        assert_relative_eq!(
            weights[&0][1],
            -9.3045547861934459,
            epsilon = TOL
        );

        assert_relative_eq!(
            weights[&1][0],
            -4.0958633027669231,
            epsilon = TOL
        );
        assert_relative_eq!(
            weights[&1][1],
            -0.4177811369331429,
            epsilon = TOL
        );
    }

    #[test]
    fn give_weights_size_check_single_target_column() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0];
        let state_weights = given_weights(&states, &col_ixs, &None);

        assert_eq!(state_weights.len(), 3);

        assert_eq!(state_weights[0].len(), 1);
        assert_eq!(state_weights[1].len(), 1);
        assert_eq!(state_weights[2].len(), 1);

        assert_eq!(state_weights[0][&0].len(), 2);
        assert_eq!(state_weights[1][&0].len(), 3);
        assert_eq!(state_weights[2][&0].len(), 2);
    }

    #[test]
    fn state_logp_values_single_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0];
        let vals = vec![vec![DType::Continuous(1.2)]];
        let logp = state_logp(&states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -2.9396185776733437, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![
            DType::Continuous(1.2),
            DType::Continuous(-0.3),
        ]];
        let logp = state_logp(&states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -4.2778895444693479, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![
            DType::Continuous(1.2),
            DType::Continuous(0.2),
        ]];
        let logp = state_logp(&states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -4.7186198999000686, epsilon = TOL);
    }

    #[test]
    fn single_state_continuous_impute_1() {
        let mut all_states = get_states_from_yaml();
        let states = vec![all_states.remove(0)];
        let x: f64 = continuous_impute(&states, 1, 0);
        assert_relative_eq!(x, 1.6831137962662617, epsilon = 10E-6);
    }

    #[test]
    fn single_state_continuous_impute_2() {
        let mut all_states = get_states_from_yaml();
        let states = vec![all_states.remove(0)];
        let x: f64 = continuous_impute(&states, 3, 0);
        assert_relative_eq!(x, -0.8244161883997966, epsilon = 10E-6);
    }

    #[test]
    fn multi_state_continuous_impute_1() {
        let mut all_states = get_states_from_yaml();
        let states = vec![all_states.remove(0), all_states.remove(0)];
        let x: f64 = continuous_impute(&states, 1, 2);
        assert_relative_eq!(x, 0.5546044921874999, epsilon = 10E-6);
    }

    #[test]
    fn multi_state_continuous_impute_2() {
        let states = get_states_from_yaml();
        let x: f64 = continuous_impute(&states, 1, 2);
        assert_relative_eq!(x, -0.2505843790156575, epsilon = 10E-6);
    }

    #[test]
    fn single_state_categorical_impute_1() {
        let state: State = get_single_categorical_state_from_yaml();
        let x: u8 = categorical_impute(&vec![state], 0, 0);
        assert_eq!(x, 2);
    }

    #[test]
    fn single_state_categorical_impute_2() {
        let state: State = get_single_categorical_state_from_yaml();
        let x: u8 = categorical_impute(&vec![state], 2, 0);
        assert_eq!(x, 0);
    }

    #[test]
    fn single_state_categorical_predict_1() {
        let state: State = get_single_categorical_state_from_yaml();
        let x: u8 = categorical_predict(&vec![state], 0, &None);
        assert_eq!(x, 2);
    }

    #[test]
    fn knn_mi_categorical_value_test_k_equal_3() {
        let xy: Vec<(f64, f64)> = vec![
            (1.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (2.0, 2.0),
            (0.0, 0.0),
        ];
        let mi = knn_mi(&xy, Some(3));
        assert_relative_eq!(mi, 0.03868290405007839, epsilon = 1E-8);
    }

    #[test]
    fn knn_mi_continuous_value_test_k_equal_3() {
        let xy: Vec<(f64, f64)> = vec![
            (0.019062707624870334, -0.29770280682399475),
            (0.0913695424229429, 0.029294692441604646),
            (0.1267316358139133, -0.5520638522640237),
            (0.9472198270311598, 0.8160802485131871),
            (1.2070213895407607, 1.4871237808112916),
            (-0.11945756610962133, 0.06193689296916297),
            (0.7988724597070684, 1.9838800018276619),
            (2.172718317219316, 1.2866016566212108),
            (-1.6439756392612739, -1.0571695383252777),
            (-0.23855091038550946, -0.5269376869509022),
        ];
        let mi = knn_mi(&xy, Some(3));
        assert_relative_eq!(mi, 0.056118242764556195, epsilon = 1E-8);
    }
}

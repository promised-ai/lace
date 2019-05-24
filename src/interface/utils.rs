use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use braid_utils::misc::{argmax, logsumexp, transpose};
use rv::dist::{Categorical, Gaussian, Mixture};
use rv::traits::{Entropy, KlDivergence, Rv};

use crate::cc::{ColModel, Datum, FType, State};
use crate::interface::Given;
use crate::optimize::fmin_bounded;

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
    states: &Vec<&State>,
    col_ixs: &Vec<usize>,
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    let mut state_weights: Vec<_> = Vec::with_capacity(states.len());

    for state in states {
        let view_weights =
            single_state_weights(&state, &col_ixs, &given, false);
        state_weights.push(view_weights);
    }
    state_weights
}

pub fn single_state_weights(
    state: &State,
    col_ixs: &Vec<usize>,
    given: &Given,
    weightless: bool,
) -> BTreeMap<usize, Vec<f64>> {
    let mut view_ixs: HashSet<usize> = HashSet::new();
    col_ixs.iter().for_each(|col_ix| {
        let view_ix = state.asgn.asgn[*col_ix];
        view_ixs.insert(view_ix);
    });

    let mut view_weights: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    view_ixs.iter().for_each(|&view_ix| {
        let weights = single_view_weights(&state, view_ix, &given, weightless);
        view_weights.insert(view_ix, weights);
    });
    view_weights
}

pub fn single_view_weights(
    state: &State,
    target_view_ix: usize,
    given: &Given,
    weightless: bool,
) -> Vec<f64> {
    let view = &state.views[target_view_ix];

    let mut weights = if weightless {
        vec![0.0; view.asgn.ncats]
    } else {
        view.asgn.log_weights()
    };

    match given {
        &Given::Conditions(ref conditions) => {
            for &(id, ref datum) in conditions {
                let in_target_view = state.asgn.asgn[id] == target_view_ix;
                if in_target_view {
                    weights = view.ftrs[&id].accum_weights(&datum, weights);
                }
            }
        }
        &Given::Nothing => (),
    }
    weights
}

// Probability calculation
// -----------------------
pub fn state_logp(
    state: &State,
    col_ixs: &Vec<usize>,
    vals: &Vec<Vec<Datum>>,
    given: &Given,
) -> Vec<f64> {
    let mut view_weights = single_state_weights(state, &col_ixs, &given, false);

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
    val: &Vec<Datum>,
    view_weights: &BTreeMap<usize, Vec<f64>>,
) -> f64 {
    // turn col_ixs and values into a given
    let given = {
        let mut obs = Vec::with_capacity(col_ixs.len());
        for (&col_ix, datum) in col_ixs.iter().zip(val) {
            obs.push((col_ix, datum.clone()));
        }
        Given::Conditions(obs)
    };

    // compute the un-normalied, 'weightless', weights using the given
    let logp_obs = single_state_weights(state, &col_ixs, &given, true);

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
        .map(|state| state.extract_continuous_cpnt(row_ix, col_ix).unwrap())
        .collect();

    if cpnts.len() == 1 {
        cpnts[0].mu
    } else {
        let f = |x: f64| {
            let logfs: Vec<f64> =
                cpnts.iter().map(|&cpnt| cpnt.ln_f(&x)).collect();
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
    let cpnts: Vec<&Categorical> = states
        .iter()
        .map(|state| state.extract_categorical_cpnt(row_ix, col_ix).unwrap())
        .collect();

    let k = cpnts[0].ln_weights.len() as u8;
    let fs: Vec<f64> = (0..k)
        .map(|x| {
            let logfs: Vec<f64> =
                cpnts.iter().map(|&cpnt| cpnt.ln_f(&x)).collect();
            logsumexp(&logfs)
        })
        .collect();
    argmax(&fs) as u8
}

pub fn categorical_entropy_single(col_ix: usize, states: &Vec<State>) -> f64 {
    let cpnt = states[0].extract_categorical_cpnt(0, col_ix).unwrap();
    let k = cpnt.ln_weights.len() as u8;

    let vals: Vec<Vec<Datum>> =
        (0..k).map(|x| vec![Datum::Categorical(x as u8)]).collect();

    let logps: Vec<Vec<f64>> = states
        .iter()
        .map(|state| state_logp(&state, &vec![col_ix], &vals, &Given::Nothing))
        .collect();

    let ln_nstates = (states.len() as f64).ln();

    transpose(&logps)
        .iter()
        .map(|lps| logsumexp(&lps) - ln_nstates)
        .fold(0.0, |acc, lp| acc - lp * lp.exp())
}

pub fn categorical_entropy_dual(
    col_a: usize,
    col_b: usize,
    states: &Vec<State>,
) -> f64 {
    let cpnt_a = states[0].extract_categorical_cpnt(0, col_a).unwrap();
    let cpnt_b = states[0].extract_categorical_cpnt(0, col_b).unwrap();

    let k_a = cpnt_a.ln_weights.len();
    let k_b = cpnt_b.ln_weights.len();

    let mut vals: Vec<Vec<Datum>> = Vec::with_capacity(k_a * k_b);
    for i in 0..k_a {
        for j in 0..k_b {
            vals.push(vec![
                Datum::Categorical(i as u8),
                Datum::Categorical(j as u8),
            ]);
        }
    }

    let logps: Vec<Vec<f64>> = states
        .iter()
        .map(|state| {
            state_logp(&state, &vec![col_a, col_b], &vals, &Given::Nothing)
        })
        .collect();

    let ln_nstates = (states.len() as f64).ln();

    transpose(&logps)
        .iter()
        .map(|lps| logsumexp(&lps) - ln_nstates)
        .fold(0.0, |acc, lp| acc - lp * lp.exp())
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
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Continuous(x)]];
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
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Categorical(x)]];
        let scores: Vec<f64> = states
            .iter()
            .map(|state| state_logp(state, &col_ixs, &y, &given)[0])
            .collect();
        logsumexp(&scores)
    };

    let k: u8 = match states[0].feature(col_ix) {
        ColModel::Categorical(ftr) => ftr.prior.symdir.k as u8,
        _ => panic!("FType mitmatch."),
    };

    let fs: Vec<f64> = (0..k).map(|x| f(x)).collect();
    argmax(&fs) as u8
}

// Predictive uncertainty helpers
// ------------------------------
trait MixtureJsd: Entropy {
    fn mix_jsd(&self) -> f64;
}

// TODO: woudl be nice to do this with traits -- to say implement for mixtures
// that implement entropy and whose components implement entropy
macro_rules! impl_mixture_jsd {
    ($fx: ty) => {
        impl MixtureJsd for Mixture<$fx> {
            fn mix_jsd(&self) -> f64 {
                let h_mixture = self.entropy();
                let h_cpnts = self
                    .weights
                    .iter()
                    .zip(self.components.iter())
                    .fold(0.0, |acc, (&w, cpnt)| acc + w * cpnt.entropy());
                h_mixture - h_cpnts
            }
        }
    };
}

impl_mixture_jsd!(Gaussian);
impl_mixture_jsd!(Categorical);

macro_rules! predunc_arm {
    ($states: expr, $col_ix: expr, $given_opt: expr, $cpnt_type: ty, $unwrap_fn: ident) => {{
        let mix_models: Vec<Mixture<$cpnt_type>> = $states
            .iter()
            .map(|state| {
                let view_ix = state.asgn.asgn[$col_ix];
                let weights =
                    single_view_weights(&state, view_ix, $given_opt, false);
                let mut mixture =
                    state.feature_as_mixture($col_ix).$unwrap_fn();
                let z = logsumexp(&weights);
                mixture.weights =
                    weights.iter().map(|w| (w - z).exp()).collect();
                mixture
            })
            .collect();
        Mixture::combine(mix_models).unwrap().mix_jsd()
    }};
}

pub fn predict_uncertainty(
    states: &Vec<State>,
    col_ix: usize,
    given: &Given,
) -> f64 {
    let ftype = {
        let view_ix = states[0].asgn.asgn[col_ix];
        states[0].views[view_ix].ftrs[&col_ix].ftype()
    };
    match ftype {
        FType::Continuous => {
            predunc_arm!(states, col_ix, &given, Gaussian, unwrap_gaussian)
        }
        FType::Categorical => {
            predunc_arm!(states, col_ix, given, Categorical, unwrap_categorical)
        }
    }
}

macro_rules! js_impunc_arm {
    ($k: expr, $row_ix: expr, $states: expr, $ftr: expr, $kind: path) => {{
        let nstates = $states.len();
        let col_ix = $ftr.id;
        let mut cpnts = Vec::with_capacity(nstates);
        cpnts.push($ftr.components[$k].fx.clone());
        for i in 1..nstates {
            let view_ix_s = $states[i].asgn.asgn[col_ix];
            let view_s = &$states[i].views[view_ix_s];
            let k_s = view_s.asgn.asgn[$row_ix];
            match &view_s.ftrs[&col_ix] {
                &$kind(ref ftr) => {
                    cpnts.push(ftr.components[k_s].fx.clone());
                }
                _ => panic!("Mismatched feature type"),
            }
        }
        Mixture::uniform(cpnts).unwrap().mix_jsd()
    }};
}

pub fn js_impute_uncertainty(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> f64 {
    let view_ix = states[0].asgn.asgn[col_ix];
    let view = &states[0].views[view_ix];
    let k = view.asgn.asgn[row_ix];
    match &view.ftrs[&col_ix] {
        &ColModel::Continuous(ref ftr) => {
            js_impunc_arm!(k, row_ix, states, ftr, ColModel::Continuous)
        }
        &ColModel::Categorical(ref ftr) => {
            js_impunc_arm!(k, row_ix, states, ftr, ColModel::Categorical)
        }
    }
}

macro_rules! kl_impunc_arm {
    ($i: expr, $ki: expr, $locators: expr, $fi: expr, $states: expr, $kind: path) => {{
        let col_ix = $fi.id;
        let mut partial_sum = 0.0;
        let cpnt_i = &$fi.components[$ki].fx;
        for (j, &(vj, kj)) in $locators.iter().enumerate() {
            if $i != j {
                let cm_j = &$states[j].views[vj].ftrs[&col_ix];
                match cm_j {
                    &$kind(ref fj) => {
                        let cpnt_j = &fj.components[kj].fx;
                        partial_sum += cpnt_i.kl(cpnt_j);
                    }
                    _ => panic!("2nd ColModel was incorrect type"),
                }
            }
        }
        partial_sum
    }};
}

pub fn kl_impute_uncertainty(
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
                kl_sum += kl_impunc_arm!(
                    i,
                    ki,
                    locators,
                    fi,
                    states,
                    ColModel::Continuous
                )
            }
            &ColModel::Categorical(ref fi) => {
                kl_sum += kl_impunc_arm!(
                    i,
                    ki,
                    locators,
                    fi,
                    states,
                    ColModel::Categorical
                )
            }
        }
    }

    let nstates = states.len() as f64;
    kl_sum / (nstates * nstates - nstates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

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

        let weights = single_view_weights(&state, 0, &Given::Nothing, false);

        assert_relative_eq!(weights[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights[1], -0.6931471805599453, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given() {
        let state = get_single_continuous_state_from_yaml();
        let given = Given::Conditions(vec![(0, Datum::Continuous(0.5))]);

        let weights = single_view_weights(&state, 0, &given, false);

        assert_relative_eq!(weights[0], -2.8570549170130315, epsilon = TOL);
        assert_relative_eq!(weights[1], -16.59893853320467, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given_weightless() {
        let state = get_single_continuous_state_from_yaml();
        let given = Given::Conditions(vec![(0, Datum::Continuous(0.5))]);

        let weights = single_view_weights(&state, 0, &given, true);

        assert_relative_eq!(weights[0], -2.1639077364530861, epsilon = TOL);
        assert_relative_eq!(weights[1], -15.905791352644725, epsilon = TOL);
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let states = get_states_from_yaml();

        let weights_0 =
            single_view_weights(&states[0], 0, &Given::Nothing, false);

        assert_relative_eq!(weights_0[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -0.6931471805599453, epsilon = TOL);

        let weights_1 =
            single_view_weights(&states[0], 1, &Given::Nothing, false);

        assert_relative_eq!(weights_1[0], -1.3862943611198906, epsilon = TOL);
        assert_relative_eq!(weights_1[1], -0.2876820724517809, epsilon = TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_one_given() {
        let states = get_states_from_yaml();

        // column 1 should not affect view 0 weights because it is assigned to
        // view 1
        let given = Given::Conditions(vec![
            (0, Datum::Continuous(0.0)),
            (1, Datum::Continuous(-1.0)),
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

        let given = Given::Conditions(vec![
            (0, Datum::Continuous(0.0)),
            (2, Datum::Continuous(-1.0)),
        ]);

        let weights_0 = single_view_weights(&states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -5.6691757676902537, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -9.3045547861934459, epsilon = TOL);
    }

    #[test]
    fn single_state_weights_value_check() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let given = Given::Conditions(vec![
            (0, Datum::Continuous(0.0)),
            (1, Datum::Continuous(-1.0)),
            (2, Datum::Continuous(-1.0)),
        ]);

        let weights = single_state_weights(&states[0], &col_ixs, &given, false);

        assert_eq!(weights.len(), 2);
        assert_eq!(weights[&0].len(), 2);
        assert_eq!(weights[&1].len(), 2);

        assert_relative_eq!(weights[&0][0], -5.6691757676902537, epsilon = TOL);
        assert_relative_eq!(weights[&0][1], -9.3045547861934459, epsilon = TOL);

        assert_relative_eq!(weights[&1][0], -4.0958633027669231, epsilon = TOL);
        assert_relative_eq!(weights[&1][1], -0.4177811369331429, epsilon = TOL);
    }

    #[test]
    fn give_weights_size_check_single_target_column() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0];
        let state_weights = given_weights(
            &states.iter().map(|s| s).collect(),
            &col_ixs,
            &Given::Nothing,
        );

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
        let vals = vec![vec![Datum::Continuous(1.2)]];
        let logp = state_logp(&states[0], &col_ixs, &vals, &Given::Nothing);

        assert_relative_eq!(logp[0], -2.9396185776733437, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(-0.3)]];
        let logp = state_logp(&states[0], &col_ixs, &vals, &Given::Nothing);

        assert_relative_eq!(logp[0], -4.2778895444693479, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(0.2)]];
        let logp = state_logp(&states[0], &col_ixs, &vals, &Given::Nothing);

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
        let x: u8 = categorical_predict(&vec![state], 0, &Given::Nothing);
        assert_eq!(x, 2);
    }
}

use std::collections::{BTreeMap, HashSet};
use std::f64::NEG_INFINITY;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use braid_stats::labeler::{Label, Labeler};
use braid_stats::{Datum, MixtureType};
use braid_utils::misc::{argmax, logsumexp, transpose};
use rv::dist::{Categorical, Gaussian, Mixture};
use rv::misc::quad;
use rv::traits::{Entropy, KlDivergence, QuadBounds, Rv};

use crate::cc::{ColModel, FType, Feature, State};
use crate::interface::Given;
use crate::optimize::fmin_bounded;

pub fn load_states<P: AsRef<Path>>(filenames: Vec<P>) -> Vec<State> {
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

/// Generate uniformly `n` distributed data for specific columns and compute
/// the reciprocal of the importance function.
pub fn gen_sobol_samples(
    col_ixs: &[usize],
    state: &State,
    n: usize,
) -> (Vec<Vec<Datum>>, f64) {
    use braid_stats::entropy::QmcEntropy;
    use braid_stats::seq::SobolSeq;

    let features: Vec<_> =
        col_ixs.iter().map(|&ix| state.feature(ix)).collect();
    let ndims: usize = features.iter().map(|ftr| ftr.ndims()).sum::<usize>();
    let halton = SobolSeq::new(ndims);

    let samples: Vec<Vec<Datum>> = halton
        .take(n)
        .map(|mut us| {
            let mut drain = us.drain(..);
            features
                .iter()
                .map(|ftr| ftr.us_to_datum(&mut drain))
                .collect()
        })
        .collect();

    let q_recip: f64 = features
        .iter()
        .fold(1_f64, |prod, ftr| prod * ftr.q_recip());

    (samples, q_recip)
}

// Weight Calculation
// ------------------
/// An enum describing whether to compute probability weights
#[derive(Debug, Clone, Copy)]
enum WeightNorm {
    /// Compute un-normalized weights
    UnNormed,
    /// Compute normalized weights
    Normed,
}

#[allow(clippy::ptr_arg)]
pub fn given_weights(
    states: &Vec<&State>,
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    let mut state_weights: Vec<_> = Vec::with_capacity(states.len());

    for state in states {
        let view_weights =
            single_state_weights(&state, &col_ixs, &given, WeightNorm::Normed);
        state_weights.push(view_weights);
    }
    state_weights
}

fn single_state_weights(
    state: &State,
    col_ixs: &[usize],
    given: &Given,
    weight_norm: WeightNorm,
) -> BTreeMap<usize, Vec<f64>> {
    let mut view_ixs: HashSet<usize> = HashSet::new();
    col_ixs.iter().for_each(|col_ix| {
        let view_ix = state.asgn.asgn[*col_ix];
        view_ixs.insert(view_ix);
    });

    let mut view_weights: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    view_ixs.iter().for_each(|&view_ix| {
        let weights = single_view_weights(&state, view_ix, &given, weight_norm);
        view_weights.insert(view_ix, weights);
    });
    view_weights
}

fn single_view_weights(
    state: &State,
    target_view_ix: usize,
    given: &Given,
    weight_norm: WeightNorm,
) -> Vec<f64> {
    let view = &state.views[target_view_ix];

    let mut weights = match weight_norm {
        WeightNorm::UnNormed => vec![0.0; view.asgn.ncats],
        WeightNorm::Normed => view.asgn.log_weights(),
    };

    match given {
        Given::Conditions(ref conditions) => {
            for &(id, ref datum) in conditions {
                let in_target_view = state.asgn.asgn[id] == target_view_ix;
                if in_target_view {
                    weights = view.ftrs[&id].accum_weights(&datum, weights);
                }
            }
        }
        Given::Nothing => (),
    }
    weights
}

// Probability calculation
// -----------------------
#[allow(clippy::ptr_arg)]
pub fn state_logp(
    state: &State,
    col_ixs: &[usize],
    vals: &Vec<Vec<Datum>>,
    given: &Given,
) -> Vec<f64> {
    let mut view_weights =
        single_state_weights(state, &col_ixs, &given, WeightNorm::Normed);

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
    col_ixs: &[usize],
    val: &[Datum],
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

    // compute the un-normalied weights using the given
    let logp_obs =
        single_state_weights(state, &col_ixs, &given, WeightNorm::UnNormed);

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
#[allow(clippy::ptr_arg)]
fn impute_bounds(states: &Vec<State>, col_ix: usize) -> (f64, f64) {
    let (lowers, uppers): (Vec<f64>, Vec<f64>) = states
        .iter()
        .map(|state| state.impute_bounds(col_ix).unwrap())
        .unzip();
    let min: f64 = lowers.iter().fold(std::f64::INFINITY, |acc, &x| x.min(acc));
    let max: f64 = uppers
        .iter()
        .fold(std::f64::NEG_INFINITY, |acc, &x| x.max(acc));
    assert!(min <= max);
    (min, max)
}

#[allow(clippy::ptr_arg)]
pub fn continuous_impute(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> f64 {
    let cpnts: Vec<Gaussian> = states
        .iter()
        .map(|state| state.component(row_ix, col_ix).into())
        .collect();

    if cpnts.len() == 1 {
        cpnts[0].mu()
    } else {
        let f = |x: f64| {
            let logfs: Vec<f64> =
                cpnts.iter().map(|cpnt| cpnt.ln_f(&x)).collect();
            -logsumexp(&logfs)
        };

        let bounds = impute_bounds(&states, col_ix);
        fmin_bounded(f, bounds, None, None)
    }
}

#[allow(clippy::ptr_arg)]
pub fn categorical_impute(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> u8 {
    let cpnts: Vec<Categorical> = states
        .iter()
        .map(|state| state.component(row_ix, col_ix).into())
        .collect();

    let k = cpnts[0].k();
    let fs: Vec<f64> = (0..k)
        .map(|x| {
            let logfs: Vec<f64> =
                cpnts.iter().map(|cpnt| cpnt.ln_f(&x)).collect();
            logsumexp(&logfs)
        })
        .collect();
    argmax(&fs) as u8
}

#[allow(clippy::ptr_arg)]
pub fn labeler_impute(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> Label {
    let cpnts: Vec<Labeler> = states
        .iter()
        .map(|state| state.component(row_ix, col_ix).into())
        .collect();

    cpnts[0]
        .support_iter()
        .fold((Label::new(0, None), NEG_INFINITY), |acc, x| {
            let logfs: Vec<f64> =
                cpnts.iter().map(|cpnt| cpnt.ln_f(&x)).collect();
            let p = logsumexp(&logfs);
            if p > acc.1 {
                (x, p)
            } else {
                acc
            }
        })
        .0
}

#[allow(clippy::ptr_arg)]
pub fn entropy_single(col_ix: usize, states: &Vec<State>) -> f64 {
    let nf = states.len() as f64;
    states
        .iter()
        .map(|state| state.feature(col_ix).to_mixture().entropy())
        .sum::<f64>()
        / nf
}

/// Joint entropy H(X, Y) where X is Categorical and Y is Gaussian
#[allow(clippy::ptr_arg)]
pub fn categorical_gaussian_entropy_dual(
    col_cat: usize,
    col_gauss: usize,
    states: &Vec<State>,
) -> f64 {
    let cat_k = {
        let cat_cpnt: Categorical = states[0].component(0, col_cat).into();
        cat_cpnt.k()
    };

    let gm: Mixture<Gaussian> = {
        let gms: Vec<MixtureType> = states
            .iter()
            .map(|state| state.feature(col_gauss).to_mixture())
            .collect();
        if let MixtureType::Gaussian(mm) = MixtureType::combine(gms) {
            mm
        } else {
            panic!("Someone this wasn't a Gaussian Mixture")
        }
    };

    let (a, b) = gm.quad_bounds();
    let col_ixs = vec![col_cat, col_gauss];

    let nf = states.len() as f64;

    (0..cat_k)
        .map(|k| {
            let x = Datum::Categorical(k as u8);
            let quad_fn = |y: f64| {
                let vals = vec![vec![x.clone(), Datum::Continuous(y)]];
                let logp = states
                    .iter()
                    .map(|state| {
                        state_logp(state, &col_ixs, &vals, &Given::Nothing)[0]
                    })
                    .sum::<f64>()
                    / nf;
                -logp * logp.exp()
            };
            quad(quad_fn, a, b)
        })
        .sum::<f64>()
}

/// Joint entropy H(X, Y) where both X and Y are Categorical
#[allow(clippy::ptr_arg)]
pub fn categorical_entropy_dual(
    col_a: usize,
    col_b: usize,
    states: &Vec<State>,
) -> f64 {
    if col_a == col_b {
        return entropy_single(col_a, states);
    }

    let cpnt_a: Categorical = states[0].component(0, col_a).into();
    let cpnt_b: Categorical = states[0].component(0, col_b).into();

    let k_a = cpnt_a.k();
    let k_b = cpnt_b.k();

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
            state_logp(&state, &[col_a, col_b], &vals, &Given::Nothing)
        })
        .collect();

    let ln_nstates = (states.len() as f64).ln();

    transpose(&logps)
        .iter()
        .map(|lps| logsumexp(&lps) - ln_nstates)
        .fold(0.0, |acc, lp| acc - lp * lp.exp())
}

pub struct MiComponents {
    /// The entropy of column a, H(A)
    pub h_a: f64,
    /// The entropy of column b, H(B)
    pub h_b: f64,
    /// The joint entropy of columns a and b, H(A, B)
    pub h_ab: f64,
}

/// Mutual information, I(X, Y), where both X and Y are Categorical
#[allow(clippy::ptr_arg)]
pub fn categorical_mi(
    col_a: usize,
    col_b: usize,
    states: &Vec<State>,
) -> MiComponents {
    let h_a = entropy_single(col_a, &states);
    if col_a == col_b {
        MiComponents {
            h_a,
            h_b: h_a,
            h_ab: h_a,
        }
    } else {
        let h_b = entropy_single(col_b, &states);
        let h_ab = categorical_entropy_dual(col_a, col_b, &states);
        MiComponents { h_a, h_b, h_ab }
    }
}

/// Mutual information, I(X, Y), where X is Categorical and Y is Gaussian
#[allow(clippy::ptr_arg)]
pub fn categorical_gaussian_mi(
    col_cat: usize,
    col_gauss: usize,
    states: &Vec<State>,
) -> MiComponents {
    let h_a = entropy_single(col_cat, &states);
    let h_b = entropy_single(col_gauss, &states);
    let h_ab = categorical_gaussian_entropy_dual(col_cat, col_gauss, &states);
    MiComponents { h_a, h_b, h_ab }
}

// Prediction
// ----------
#[allow(clippy::ptr_arg)]
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

#[allow(clippy::ptr_arg)]
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
        ColModel::Categorical(ftr) => ftr.prior.symdir.k() as u8,
        _ => panic!("FType mitmatch."),
    };

    let fs: Vec<f64> = (0..k).map(f).collect();
    argmax(&fs) as u8
}

// XXX: Not 100% sure how to predict `label` given `truth'. For now, we're
// going to predict (label, truth), given other columns.
#[allow(clippy::ptr_arg)]
pub fn labeler_predict(
    states: &Vec<State>,
    col_ix: usize,
    given: &Given,
) -> Label {
    let col_ixs: Vec<usize> = vec![col_ix];

    let f = |x: Label| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Label(x)]];
        let scores: Vec<f64> = states
            .iter()
            .map(|state| state_logp(state, &col_ixs, &y, &given)[0])
            .collect();
        logsumexp(&scores)
    };

    let labeler: &Labeler = match states[0].feature(col_ix) {
        ColModel::Labeler(ftr) => &ftr.components[0].fx,
        _ => panic!("FType mitmatch."),
    };

    labeler
        .support_iter()
        .fold((Label::new(0, None), NEG_INFINITY), |acc, x| {
            let p = f(x);
            if p > acc.1 {
                (x, p)
            } else {
                acc
            }
        })
        .0
}

// Predictive uncertainty helpers
// ------------------------------

// Jensen-shannon-divergence for a mixture
fn jsd<Fx>(mm: Mixture<Fx>) -> f64
where
    MixtureType: From<Mixture<Fx>>,
    Fx: Entropy + Clone + std::fmt::Debug + PartialOrd,
{
    let h_cpnts = mm
        .weights()
        .iter()
        .zip(mm.components().iter())
        .fold(0.0, |acc, (&w, cpnt)| w.mul_add(cpnt.entropy(), acc));

    let mt: MixtureType = mm.into();
    let h_mixture = mt.entropy();

    h_mixture - h_cpnts
}

macro_rules! predunc_arm {
    ($states: expr, $col_ix: expr, $given_opt: expr, $cpnt_type: ty) => {{
        let mix_models: Vec<Mixture<$cpnt_type>> = $states
            .iter()
            .map(|state| {
                let view_ix = state.asgn.asgn[$col_ix];
                let weights = single_view_weights(
                    &state,
                    view_ix,
                    $given_opt,
                    WeightNorm::Normed,
                );

                let mut mixture: Mixture<$cpnt_type> =
                    state.feature($col_ix).to_mixture().into();

                let z = logsumexp(&weights);

                // FIXME: need setters in rv so we don't have to re-init and clone so much
                let new_weights = weights.iter().map(|w| (w - z).exp()).collect();
                mixture = Mixture::new(new_weights, mixture.components().to_owned()).unwrap();

                mixture
            })
            .collect();
        let mm = Mixture::combine(mix_models);
        jsd(mm)
    }};
}

#[allow(clippy::ptr_arg)]
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
        FType::Continuous => predunc_arm!(states, col_ix, &given, Gaussian),
        FType::Categorical => predunc_arm!(states, col_ix, given, Categorical),
        FType::Labeler => predunc_arm!(states, col_ix, given, Labeler),
    }
}

macro_rules! js_impunc_arm {
    ($k: expr, $row_ix: expr, $states: expr, $ftr: expr, $variant: ident) => {{
        let nstates = $states.len();
        let col_ix = $ftr.id;
        let mut cpnts = Vec::with_capacity(nstates);
        cpnts.push($ftr.components[$k].fx.clone());
        for i in 1..nstates {
            let view_ix_s = $states[i].asgn.asgn[col_ix];
            let view_s = &$states[i].views[view_ix_s];
            let k_s = view_s.asgn.asgn[$row_ix];
            match &view_s.ftrs[&col_ix] {
                ColModel::$variant(ref ftr) => {
                    cpnts.push(ftr.components[k_s].fx.clone());
                }
                _ => panic!("Mismatched feature type"),
            }
        }
        jsd(Mixture::uniform(cpnts).unwrap())
    }};
}

#[allow(clippy::ptr_arg)]
pub fn js_impute_uncertainty(
    states: &Vec<State>,
    row_ix: usize,
    col_ix: usize,
) -> f64 {
    let view_ix = states[0].asgn.asgn[col_ix];
    let view = &states[0].views[view_ix];
    let k = view.asgn.asgn[row_ix];
    match &view.ftrs[&col_ix] {
        ColModel::Continuous(ref ftr) => {
            js_impunc_arm!(k, row_ix, states, ftr, Continuous)
        }
        ColModel::Categorical(ref ftr) => {
            js_impunc_arm!(k, row_ix, states, ftr, Categorical)
        }
        ColModel::Labeler(ref ftr) => {
            js_impunc_arm!(k, row_ix, states, ftr, Labeler)
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
                    $kind(ref fj) => {
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

#[allow(clippy::ptr_arg)]
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
            ColModel::Continuous(ref fi) => {
                kl_sum += kl_impunc_arm!(
                    i,
                    ki,
                    locators,
                    fi,
                    states,
                    ColModel::Continuous
                )
            }
            ColModel::Categorical(ref fi) => {
                kl_sum += kl_impunc_arm!(
                    i,
                    ki,
                    locators,
                    fi,
                    states,
                    ColModel::Categorical
                )
            }
            ColModel::Labeler(ref fi) => {
                kl_sum += kl_impunc_arm!(
                    i,
                    ki,
                    locators,
                    fi,
                    states,
                    ColModel::Labeler
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

    fn get_single_labeler_state_from_yaml() -> State {
        let filenames = vec!["resources/test/single-labeler.yaml"];
        load_states(filenames).remove(0)
    }

    fn get_states_from_yaml() -> Vec<State> {
        let filenames = vec![
            "resources/test/small/small-state-1.yaml",
            "resources/test/small/small-state-2.yaml",
            "resources/test/small/small-state-3.yaml",
        ];
        load_states(filenames)
    }

    fn get_entropy_states_from_yaml() -> Vec<State> {
        let filenames = vec![
            "resources/test/entropy/entropy-state-1.yaml",
            "resources/test/entropy/entropy-state-2.yaml",
        ];
        load_states(filenames)
    }

    #[test]
    fn single_continuous_column_weights_no_given() {
        let state = get_single_continuous_state_from_yaml();

        let weights =
            single_view_weights(&state, 0, &Given::Nothing, WeightNorm::Normed);

        assert_relative_eq!(weights[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights[1], -0.6931471805599453, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given() {
        let state = get_single_continuous_state_from_yaml();
        let given = Given::Conditions(vec![(0, Datum::Continuous(0.5))]);

        let weights =
            single_view_weights(&state, 0, &given, WeightNorm::Normed);

        assert_relative_eq!(weights[0], -2.8570549170130315, epsilon = TOL);
        assert_relative_eq!(weights[1], -16.59893853320467, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given_weightless() {
        let state = get_single_continuous_state_from_yaml();
        let given = Given::Conditions(vec![(0, Datum::Continuous(0.5))]);

        let weights =
            single_view_weights(&state, 0, &given, WeightNorm::UnNormed);

        assert_relative_eq!(weights[0], -2.1639077364530861, epsilon = TOL);
        assert_relative_eq!(weights[1], -15.905791352644725, epsilon = TOL);
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let states = get_states_from_yaml();

        let weights_0 = single_view_weights(
            &states[0],
            0,
            &Given::Nothing,
            WeightNorm::Normed,
        );

        assert_relative_eq!(weights_0[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -0.6931471805599453, epsilon = TOL);

        let weights_1 = single_view_weights(
            &states[0],
            1,
            &Given::Nothing,
            WeightNorm::Normed,
        );

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

        let weights_0 =
            single_view_weights(&states[0], 0, &given, WeightNorm::Normed);

        assert_relative_eq!(weights_0[0], -3.1589583681201292, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -1.9265784475169849, epsilon = TOL);

        let weights_1 =
            single_view_weights(&states[0], 1, &given, WeightNorm::Normed);

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

        let weights_0 =
            single_view_weights(&states[0], 0, &given, WeightNorm::Normed);

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

        let weights = single_state_weights(
            &states[0],
            &col_ixs,
            &given,
            WeightNorm::Normed,
        );

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

    #[test]
    fn single_state_categorical_entropy() {
        let state: State = get_single_categorical_state_from_yaml();
        let h = entropy_single(0, &vec![state]);
        assert_relative_eq!(h, 1.36854170815232, epsilon = 10E-6);
    }

    #[test]
    fn single_state_categorical_self_entropy() {
        let state: State = get_single_categorical_state_from_yaml();
        let states = vec![state];
        let h_x = entropy_single(0, &states);
        let h_xx = categorical_entropy_dual(0, 0, &states);
        assert_relative_eq!(h_xx, h_x, epsilon = 10E-6);
    }

    #[test]
    fn single_state_labeler_impute_2() {
        let state: State = get_single_labeler_state_from_yaml();
        let x: Label = labeler_impute(&vec![state], 9, 0);
        assert_eq!(
            x,
            Label {
                label: 0,
                truth: Some(0)
            }
        );
    }

    #[test]
    fn single_state_labeler_impute_1() {
        let state: State = get_single_labeler_state_from_yaml();
        let x: Label = labeler_impute(&vec![state], 1, 0);
        assert_eq!(
            x,
            Label {
                label: 1,
                truth: Some(1)
            }
        );
    }

    #[test]
    fn single_state_dual_categorical_entropy_0() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.drain(..).next().unwrap();
        let hxy = categorical_entropy_dual(2, 3, &vec![state]);
        assert_relative_eq!(hxy, 2.0503963193592734, epsilon = 1E-14);
    }

    #[test]
    fn single_state_dual_categorical_entropy_1() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.pop().unwrap();
        let hxy = categorical_entropy_dual(2, 3, &vec![state]);
        assert_relative_eq!(hxy, 2.035433971709626, epsilon = 1E-14);
    }

    #[test]
    fn multi_state_dual_categorical_entropy_1() {
        let states = get_entropy_states_from_yaml();
        let hxy = categorical_entropy_dual(2, 3, &states);
        assert_relative_eq!(hxy, 2.0504022456286415, epsilon = 1E-14);
    }

    #[test]
    fn single_state_categorical_gaussian_entropy_0() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.drain(..).next().unwrap();
        let hxy = categorical_gaussian_entropy_dual(2, 0, &vec![state]);
        assert_relative_eq!(hxy, 2.726163712601034, epsilon = 1E-9);
    }

    #[test]
    fn single_state_categorical_gaussian_entropy_1() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.pop().unwrap();
        let hxy = categorical_gaussian_entropy_dual(2, 0, &vec![state]);
        assert_relative_eq!(hxy, 2.7354575323710746, epsilon = 1E-9);
    }

    #[test]
    fn sobol_samples() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.pop().unwrap();
        let (samples, _) = gen_sobol_samples(&vec![0, 2, 3], &state, 102);

        assert_eq!(samples.len(), 102);

        for vals in samples {
            assert_eq!(vals.len(), 3);
            assert!(vals[0].is_continuous());
            assert!(vals[1].is_categorical());
            assert!(vals[2].is_categorical());
        }
    }

    fn sobolo_vs_exact_entropy(col_ix: usize, n: usize) -> (f64, f64) {
        let mut states = get_entropy_states_from_yaml();
        let state = states.pop().unwrap();

        let h_sobol = {
            let (samples, q_recip) =
                gen_sobol_samples(&vec![col_ix], &state, n);
            let logps =
                state_logp(&state, &vec![col_ix], &samples, &Given::Nothing);

            let h: f64 = logps.iter().map(|logp| -logp * logp.exp()).sum();

            h * q_recip / (n as f64)
        };

        let h_exact = entropy_single(col_ix, &vec![state]);

        (h_exact, h_sobol)
    }

    #[test]
    #[ignore]
    fn sobol_single_categorical_entropy_vs_exact() {
        // TODO: This numbers don't line up very well
        let (h_exact, h_sobol) = sobolo_vs_exact_entropy(2, 10_000);
        println!("Exact: {}, Sobol: {}", h_exact, h_sobol);
    }

    #[test]
    fn sobol_single_gaussian_entropy_vs_exact() {
        let (h_exact, h_sobol) = sobolo_vs_exact_entropy(0, 10_000);
        assert_relative_eq!(h_exact, h_sobol, epsilon = 1E-8);
    }
}

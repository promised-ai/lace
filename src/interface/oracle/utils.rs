use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::f64::{INFINITY, NEG_INFINITY};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use braid_stats::labeler::{Label, Labeler};
use braid_stats::{Datum, MixtureType};
use braid_utils::{argmax, logsumexp, transpose};
use rv::dist::{Categorical, Gaussian, Mixture, Poisson};
use rv::misc::quad;
use rv::traits::{Entropy, KlDivergence, QuadBounds, Rv};

use crate::cc::{ColModel, FType, Feature, State};
use crate::interface::Given;
use crate::optimize::{fmin_bounded, fmin_brute};

/// Generates samples
pub struct Simulator<'s, R: rand::Rng> {
    rng: &'s mut R,
    /// A list of the states
    states: &'s Vec<&'s State>,
    /// The view weights for each state
    weights: &'s Vec<BTreeMap<usize, Vec<f64>>>,
    /// Draws state indices at uniform
    state_ixer: Categorical,
    /// List of state indices from which to simulate
    state_ixs: Vec<usize>,
    /// List of state indices from which to simulate
    col_ixs: &'s [usize],
    component_ixers: BTreeMap<usize, Vec<Categorical>>,
}

impl<'s, R: rand::Rng> Simulator<'s, R> {
    pub fn new(
        states: &'s Vec<&'s State>,
        weights: &'s Vec<BTreeMap<usize, Vec<f64>>>,
        state_ixs: Option<Vec<usize>>,
        col_ixs: &'s [usize],
        rng: &'s mut R,
    ) -> Self {
        Simulator {
            rng,
            weights,
            state_ixer: match state_ixs {
                Some(ref ixs) => Categorical::uniform(ixs.len()),
                None => Categorical::uniform(states.len()),
            },
            state_ixs: match state_ixs {
                Some(ixs) => ixs,
                None => (0..states.len()).collect(),
            },
            states,
            col_ixs,
            component_ixers: BTreeMap::new(),
        }
    }
}

impl<'s, R: rand::Rng> Iterator for Simulator<'s, R> {
    type Item = Vec<Datum>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = &mut self.rng;

        // choose a random state
        let draw_ix: usize = self.state_ixer.draw(&mut rng);
        let state_ix: usize = self.state_ixs[draw_ix];
        let state = &self.states[draw_ix];

        let weights = &self.weights;

        // for each view
        //   choose a random component from the weights
        self.component_ixers.entry(state_ix).or_insert_with(|| {
            // TODO: use Categorical::new_unchecked when rv 0.9.3 drops.
            // from_ln_weights checks that the input logsumexp's to 0
            weights[draw_ix]
                .values()
                .map(|view_weights| {
                    Categorical::from_ln_weights(view_weights.clone()).unwrap()
                })
                .collect()
        });

        let cpnt_ixs: BTreeMap<usize, usize> = self.weights[draw_ix]
            .keys()
            .zip(self.component_ixers[&state_ix].iter())
            .map(|(&view_ix, cpnt_ixer)| (view_ix, cpnt_ixer.draw(&mut rng)))
            .collect();

        let xs: Vec<_> = self
            .col_ixs
            .iter()
            .map(|col_ix| {
                let view_ix = state.asgn.asgn[*col_ix];
                let k = cpnt_ixs[&view_ix];
                state.views[view_ix].ftrs[col_ix].draw(k, &mut rng)
            })
            .collect();

        Some(xs)
    }
}

/// Computes probabilities from streams of data
pub struct Calcultor<'s, Xs>
where
    Xs: Iterator,
    Xs::Item: Borrow<Vec<Datum>>,
{
    /// A list of the states
    states: &'s Vec<&'s State>,
    /// The view weights for each state
    weights: &'s Vec<BTreeMap<usize, Vec<f64>>>,
    /// List of state indices from which to simulate
    col_ixs: &'s [usize],
    values: &'s mut Xs,
}

impl<'s, Xs> Calcultor<'s, Xs>
where
    Xs: Iterator,
    Xs::Item: Borrow<Vec<Datum>>,
{
    pub fn new(
        values: &'s mut Xs,
        states: &'s Vec<&'s State>,
        weights: &'s Vec<BTreeMap<usize, Vec<f64>>>,
        col_ixs: &'s [usize],
    ) -> Self {
        Calcultor {
            values,
            weights,
            states,
            col_ixs,
        }
    }
}

impl<'s, Xs> Iterator for Calcultor<'s, Xs>
where
    Xs: Iterator,
    Xs::Item: Borrow<Vec<Datum>>,
{
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        match self.values.next() {
            Some(xs) => {
                let ln_n = (self.states.len() as f64).ln();
                let col_ixs = self.col_ixs;
                let logps = self
                    .states
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(state, weights)| {
                        single_val_logp(
                            state,
                            col_ixs,
                            xs.borrow(),
                            weights.clone(),
                        )
                    })
                    .collect::<Vec<f64>>();
                Some(logsumexp(&logps) - ln_n)
            }
            None => None,
        }
    }
}

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
    use braid_stats::seq::SobolSeq;
    use braid_stats::QmcEntropy;

    let features: Vec<_> =
        col_ixs.iter().map(|&ix| state.feature(ix)).collect();
    let us_needed: usize = features.iter().map(|ftr| ftr.us_needed()).sum();
    let sobol = SobolSeq::new(us_needed);

    let samples: Vec<Vec<Datum>> = sobol
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
#[allow(clippy::ptr_arg)]
#[inline]
pub fn given_weights(
    states: &Vec<&State>,
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    let mut state_weights: Vec<_> = Vec::with_capacity(states.len());

    for state in states {
        let view_weights = single_state_weights(&state, &col_ixs, &given);
        state_weights.push(view_weights);
    }
    state_weights
}

#[inline]
pub fn state_weights(
    states: &[State],
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    states
        .iter()
        .map(|state| single_state_weights(state, col_ixs, given))
        .collect()
}

#[inline]
pub fn single_state_weights(
    state: &State,
    col_ixs: &[usize],
    given: &Given,
) -> BTreeMap<usize, Vec<f64>> {
    let mut view_weights: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    col_ixs
        .iter()
        .map(|&ix| state.asgn.asgn[ix])
        .for_each(|view_ix| {
            if !view_weights.contains_key(&view_ix) {
                let weights = single_view_weights(&state, view_ix, &given);
                view_weights.insert(view_ix, weights);
            }
        });

    view_weights
}

#[inline]
fn single_view_weights(
    state: &State,
    target_view_ix: usize,
    given: &Given,
) -> Vec<f64> {
    let view = &state.views[target_view_ix];
    let mut weights = view.weights.iter().map(|w| w.ln()).collect();

    match given {
        Given::Conditions(ref conditions) => {
            for &(id, ref datum) in conditions {
                let in_target_view = state.asgn.asgn[id] == target_view_ix;
                if in_target_view {
                    view.ftrs[&id].accum_weights(&datum, &mut weights);
                }
            }
            let z = logsumexp(&weights);
            weights.iter_mut().for_each(|w| *w -= z);
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
    view_weights_opt: Option<&BTreeMap<usize, Vec<f64>>>,
) -> Vec<f64> {
    match view_weights_opt {
        Some(view_weights) => vals
            .iter()
            .map(|val| {
                single_val_logp(&state, &col_ixs, &val, view_weights.clone())
            })
            .collect(),
        None => {
            let mut view_weights =
                single_state_weights(state, &col_ixs, &given);

            // normalize view weights
            for weights in view_weights.values_mut() {
                let logz = logsumexp(weights);
                weights.iter_mut().for_each(|w| *w -= logz);
            }
            vals.iter()
                .map(|val| {
                    single_val_logp(
                        &state,
                        &col_ixs,
                        &val,
                        view_weights.clone(),
                    )
                })
                .collect()
        }
    }
}

fn single_val_logp(
    state: &State,
    col_ixs: &[usize],
    val: &[Datum],
    mut view_weights: BTreeMap<usize, Vec<f64>>,
) -> f64 {
    // TODO: is there a way to do this without cloning the view_weights?
    col_ixs
        .iter()
        .zip(val)
        .map(|(col_ix, datum)| (col_ix, state.asgn.asgn[*col_ix], datum))
        .for_each(|(col_ix, view_ix, datum)| {
            state.views[view_ix].ftrs[col_ix]
                .accum_weights(datum, view_weights.get_mut(&view_ix).unwrap());
            // let weights = ;
            // view_weights.insert(
            //     view_ix,
            //     state.views[view_ix].ftrs[col_ix].accum_weights(datum, weights),
            // );
        });

    view_weights.values().map(|logps| logsumexp(logps)).sum()
}
// Imputation
// ----------
#[allow(clippy::ptr_arg)]
fn impute_bounds(states: &Vec<State>, col_ix: usize) -> (f64, f64) {
    states
        .iter()
        .map(|state| state.impute_bounds(col_ix).unwrap())
        .fold((INFINITY, NEG_INFINITY), |(min, max), (lower, upper)| {
            (min.min(lower), max.max(upper))
        })
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
        let n_grid = 100;
        let step_size = (bounds.1 - bounds.0) / (n_grid as f64);
        let x0 = fmin_brute(&f, bounds, n_grid);
        fmin_bounded(f, (x0 - step_size, x0 + step_size), None, None)
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
pub fn count_impute(states: &Vec<State>, row_ix: usize, col_ix: usize) -> u32 {
    use braid_utils::MinMax;
    use rv::traits::Mean;

    let cpnts: Vec<Poisson> = states
        .iter()
        .map(|state| state.component(row_ix, col_ix).into())
        .collect();

    let (lower, upper) = {
        let (lower, upper) = cpnts
            .iter()
            .map(|cpnt| {
                let mean: f64 = cpnt.mean().expect("Poisson always has a mean");
                mean
            })
            .minmax()
            .unwrap();
        ((lower.ceil() - 1.0) as u32, upper.floor() as u32)
    };

    // use fx instead of x so we can sum in place and not worry about
    // allocating a vector. Since there is inly one number in the likelihood,
    // we shouldn't have numerical issues.
    let fx = |x: u32| cpnts.iter().map(|cpnt| cpnt.f(&x)).sum::<f64>();

    (lower..=upper)
        .skip(1)
        .fold((lower, fx(lower)), |(argmax, max), xi| {
            let fxi = fx(xi);
            if fxi > max {
                (xi, fxi)
            } else {
                (argmax, max)
            }
        })
        .0
}

#[allow(clippy::ptr_arg)]
pub fn entropy_single(col_ix: usize, states: &Vec<State>) -> f64 {
    let mixtures = states
        .iter()
        .map(|state| state.feature_as_mixture(col_ix))
        .collect();
    let mixture = MixtureType::combine(mixtures);
    mixture.entropy()
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
            .map(|state| state.feature_as_mixture(col_gauss))
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
    let log_nstates = nf.ln();

    let state_weights = state_weights(&states, &col_ixs, &Given::Nothing);

    (0..cat_k)
        .map(|k| {
            let x = Datum::Categorical(k as u8);

            let quad_fn = |y: f64| {
                let vals = vec![vec![x.clone(), Datum::Continuous(y)]];
                let logp = {
                    let logps: Vec<f64> = states
                        .iter()
                        .zip(state_weights.iter())
                        .map(|(state, view_weights)| {
                            state_logp(
                                state,
                                &col_ixs,
                                &vals,
                                &Given::Nothing,
                                Some(view_weights),
                            )[0]
                        })
                        .collect();
                    logsumexp(&logps) - log_nstates
                };
                -logp * logp.exp()
            };
            quad(quad_fn, a, b)
        })
        .sum::<f64>()
}

/// Computes entropy among categorical columns exactly via enumeration
pub fn categorical_joint_entropy(col_ixs: &[usize], states: &[State]) -> f64 {
    let ranges = col_ixs
        .iter()
        .map(|&ix| {
            let cpnt: Categorical = states[0].component(0, ix).into();
            cpnt.k() as u8
        })
        .collect();

    let vals = braid_utils::CategoricalCartProd::new(ranges)
        .map(|mut xs| {
            let vals: Vec<_> = xs.drain(..).map(Datum::Categorical).collect();
            vals
        })
        .collect();

    // TODO: this is a pattern that appears a lot. I should DRY it.
    let logps: Vec<Vec<f64>> = states
        .iter()
        .map(|state| state_logp(&state, col_ixs, &vals, &Given::Nothing, None))
        .collect();

    let ln_nstates = (states.len() as f64).ln();

    transpose(&logps)
        .iter()
        .map(|lps| logsumexp(&lps) - ln_nstates)
        .fold(0.0, |acc, lp| acc - lp * lp.exp())
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
            state_logp(&state, &[col_a, col_b], &vals, &Given::Nothing, None)
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
#[allow(clippy::ptr_arg)]
pub fn continuous_predict(
    states: &Vec<State>,
    col_ix: usize,
    given: &Given,
) -> f64 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let state_weights = state_weights(&states, &col_ixs, given);

    let f = |x: f64| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Continuous(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, &given, Some(&view_weights))[0]
            })
            .collect();
        -logsumexp(&scores)
    };

    let bounds = impute_bounds(&states, col_ix);
    let n_grid = 100;
    let step_size = (bounds.1 - bounds.0) / (n_grid as f64);
    let x0 = fmin_brute(&f, bounds, n_grid);
    fmin_bounded(f, (x0 - step_size, x0 + step_size), None, None)
}

#[allow(clippy::ptr_arg)]
pub fn categorical_predict(
    states: &Vec<State>,
    col_ix: usize,
    given: &Given,
) -> u8 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let state_weights = state_weights(&states, &col_ixs, given);

    let f = |x: u8| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Categorical(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, &given, Some(&view_weights))[0]
            })
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

    let state_weights = state_weights(&states, &col_ixs, given);

    let f = |x: Label| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Label(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, &given, Some(&view_weights))[0]
            })
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

#[allow(clippy::ptr_arg)]
pub fn count_predict(states: &Vec<State>, col_ix: usize, given: &Given) -> u32 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let state_weights = state_weights(&states, &col_ixs, &given);

    let ln_fx = |x: u32| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Count(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, &given, Some(&view_weights))[0]
            })
            .collect();
        logsumexp(&scores)
    };

    let (lower, upper) = {
        let (lower, upper) = impute_bounds(&states, col_ix);
        ((lower + 0.5) as u32, (upper + 0.5) as u32)
    };

    (lower..=upper)
        .skip(1)
        .fold((lower, ln_fx(lower)), |(argmax, max), xi| {
            let ln_fxi = ln_fx(xi);
            if ln_fxi > max {
                (xi, ln_fxi)
            } else {
                (argmax, max)
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
    Fx: Entropy + Clone + std::fmt::Debug,
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
                let weights = single_view_weights(&state, view_ix, $given_opt);

                let mut mixture: Mixture<$cpnt_type> =
                    state.feature_as_mixture($col_ix).into();

                let z = logsumexp(&weights);

                let new_weights =
                    weights.iter().map(|w| (w - z).exp()).collect();
                mixture.set_weights_unchecked(new_weights);

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
        FType::Continuous => predunc_arm!(states, col_ix, given, Gaussian),
        FType::Categorical => predunc_arm!(states, col_ix, given, Categorical),
        FType::Labeler => predunc_arm!(states, col_ix, given, Labeler),
        FType::Count => predunc_arm!(states, col_ix, given, Poisson),
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
        ColModel::Count(ref ftr) => {
            js_impunc_arm!(k, row_ix, states, ftr, Count)
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
            ColModel::Count(ref fi) => {
                kl_sum +=
                    kl_impunc_arm!(i, ki, locators, fi, states, ColModel::Count)
            }
        }
    }

    let nstates = states.len() as f64;
    kl_sum / (nstates * nstates - nstates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OracleT;
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

    fn get_single_count_state_from_yaml() -> State {
        let filenames = vec!["resources/test/single-count.yaml"];
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

    pub fn old_categorical_entropy_single(
        col_ix: usize,
        states: &Vec<State>,
    ) -> f64 {
        let cpnt: Categorical = states[0].component(0, col_ix).into();
        let k = cpnt.k();

        let mut vals: Vec<Vec<Datum>> = Vec::with_capacity(k);
        for i in 0..k {
            vals.push(vec![Datum::Categorical(i as u8)]);
        }

        let logps: Vec<Vec<f64>> = states
            .iter()
            .map(|state| {
                state_logp(&state, &[col_ix], &vals, &Given::Nothing, None)
            })
            .collect();

        let ln_nstates = (states.len() as f64).ln();

        transpose(&logps)
            .iter()
            .map(|lps| logsumexp(&lps) - ln_nstates)
            .fold(0.0, |acc, lp| acc - (lp * lp.exp()))
    }

    #[test]
    fn single_continuous_column_weights_no_given() {
        let state = get_single_continuous_state_from_yaml();

        let weights = single_view_weights(&state, 0, &Given::Nothing);

        assert_relative_eq!(weights[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights[1], -0.6931471805599453, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given() {
        let state = get_single_continuous_state_from_yaml();
        let given = Given::Conditions(vec![(0, Datum::Continuous(0.5))]);

        let weights = single_view_weights(&state, 0, &given);
        let target = {
            let mut unnormed_targets =
                vec![-2.8570549170130315, -16.59893853320467];
            let z = logsumexp(&unnormed_targets);
            unnormed_targets.iter_mut().for_each(|w| *w -= z);
            unnormed_targets
        };

        assert_relative_eq!(weights[0], target[0], epsilon = TOL);
        assert_relative_eq!(weights[1], target[1], epsilon = TOL);
    }

    #[test]
    fn continuous_predict_with_spread_out_modes() {
        let states = {
            let filenames =
                vec!["resources/test/spread-out-continuous-modes.yaml"];
            load_states(filenames)
        };

        let x = continuous_predict(&states, 0, &Given::Nothing);
        assert_relative_eq!(x, -0.12, epsilon = 1E-5);
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let states = get_states_from_yaml();

        let weights_0 = single_view_weights(&states[0], 0, &Given::Nothing);

        assert_relative_eq!(weights_0[0], -0.6931471805599453, epsilon = TOL);
        assert_relative_eq!(weights_0[1], -0.6931471805599453, epsilon = TOL);

        let weights_1 = single_view_weights(&states[0], 1, &Given::Nothing);

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

        let weights_0 = single_view_weights(&states[0], 0, &given);
        let weights_1 = single_view_weights(&states[0], 1, &given);
        {
            let unnormed_targets =
                vec![-3.1589583681201292, -1.9265784475169849];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights_0[0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights_0[1], targets[1], epsilon = TOL);
        }

        {
            let unnormed_targets =
                vec![-4.0958633027669231, -0.4177811369331429];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights_1[0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights_1[1], targets[1], epsilon = TOL);
        }
    }

    #[test]
    fn single_view_weights_state_0_with_added_given() {
        let states = get_states_from_yaml();

        let given = Given::Conditions(vec![
            (0, Datum::Continuous(0.0)),
            (2, Datum::Continuous(-1.0)),
        ]);

        let weights_0 = single_view_weights(&states[0], 0, &given);

        {
            let unnormed_targets =
                vec![-5.6691757676902537, -9.3045547861934459];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights_0[0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights_0[1], targets[1], epsilon = TOL);
        }
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

        let weights = single_state_weights(&states[0], &col_ixs, &given);

        assert_eq!(weights.len(), 2);
        assert_eq!(weights[&0].len(), 2);
        assert_eq!(weights[&1].len(), 2);

        {
            let unnormed_targets =
                vec![-5.6691757676902537, -9.3045547861934459];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights[&0][0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights[&0][1], targets[1], epsilon = TOL);
        }

        {
            let unnormed_targets =
                vec![-4.0958633027669231, -0.4177811369331429];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights[&1][0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights[&1][1], targets[1], epsilon = TOL);
        }
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
        let logp =
            state_logp(&states[0], &col_ixs, &vals, &Given::Nothing, None);

        assert_relative_eq!(logp[0], -2.9396185776733437, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(-0.3)]];
        let logp =
            state_logp(&states[0], &col_ixs, &vals, &Given::Nothing, None);

        assert_relative_eq!(logp[0], -4.2778895444693479, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view_precomp() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(-0.3)]];
        let view_weights =
            single_state_weights(&states[0], &col_ixs, &Given::Nothing);

        let logp = state_logp(
            &states[0],
            &col_ixs,
            &vals,
            &Given::Nothing,
            Some(&view_weights),
        );

        assert_relative_eq!(logp[0], -4.2778895444693479, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(0.2)]];
        let logp =
            state_logp(&states[0], &col_ixs, &vals, &Given::Nothing, None);

        assert_relative_eq!(logp[0], -4.7186198999000686, epsilon = TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view_precomp() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(0.2)]];
        let view_weights =
            single_state_weights(&states[0], &col_ixs, &Given::Nothing);
        let logp = state_logp(
            &states[0],
            &col_ixs,
            &vals,
            &Given::Nothing,
            Some(&view_weights),
        );

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
        assert_relative_eq!(h_xx, h_x, epsilon = 1E-12);
    }

    #[test]
    fn multi_state_categorical_self_entropy() {
        let state: State = get_single_categorical_state_from_yaml();
        let states = vec![state];
        let h_x = entropy_single(0, &states);
        let h_xx = categorical_entropy_dual(0, 0, &states);
        assert_relative_eq!(h_xx, h_x, epsilon = 1E-12);
    }

    #[test]
    fn multi_state_categorical_single_entropy() {
        let states = get_entropy_states_from_yaml();
        let h_x = entropy_single(2, &states);
        assert_relative_eq!(h_x, 1.3687155004671951, epsilon = 1E-12);
    }

    #[test]
    fn multi_state_categorical_single_entropy_vs_old() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        for col_ix in 0..oracle.ncols() {
            let h_x_new = entropy_single(col_ix, &oracle.states);
            let h_x_old =
                old_categorical_entropy_single(col_ix, &oracle.states);
            assert_relative_eq!(h_x_new, h_x_old, epsilon = 1E-12);
        }
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
    fn single_state_count_impute_1() {
        let states = vec![get_single_count_state_from_yaml()];
        let x: u32 = count_impute(&states, 1, 0);
        assert_eq!(x, 1);
    }

    #[test]
    fn single_state_count_impute_2() {
        let states = vec![get_single_count_state_from_yaml()];
        let x: u32 = count_impute(&states, 1, 0);
        assert_eq!(x, 1);
    }

    #[test]
    fn single_state_count_predict() {
        let states = vec![get_single_count_state_from_yaml()];
        let x: u32 = count_predict(&states, 0, &Given::Nothing);
        assert_eq!(x, 1);
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
    fn single_state_dual_categorical_entropy_vs_joint_equiv() {
        let states = {
            let mut states = get_entropy_states_from_yaml();
            let state = states.pop().unwrap();
            vec![state]
        };
        let hxy_dual = categorical_entropy_dual(2, 3, &states);
        let hxy_joint = categorical_joint_entropy(&vec![2, 3], &states);

        assert_relative_eq!(hxy_dual, hxy_joint, epsilon = 1E-14);
    }

    #[test]
    fn multi_state_dual_categorical_entropy_1() {
        let states = get_entropy_states_from_yaml();
        let hxy = categorical_entropy_dual(2, 3, &states);
        assert_relative_eq!(hxy, 2.0504022456286415, epsilon = 1E-14);
    }

    #[test]
    fn multi_state_dual_categorical_entropy_vs_joint_equiv() {
        let states = get_entropy_states_from_yaml();
        let hxy_dual = categorical_entropy_dual(2, 3, &states);
        let hxy_joint = categorical_joint_entropy(&vec![2, 3], &states);
        assert_relative_eq!(hxy_dual, hxy_joint, epsilon = 1E-14);
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
    fn multi_state_categorical_gaussian_entropy_0() {
        let states = get_entropy_states_from_yaml();
        let hxy = categorical_gaussian_entropy_dual(2, 0, &states);
        assert_relative_eq!(hxy, 2.744356173055859, epsilon = 1E-8);
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
            let logps = state_logp(
                &state,
                &vec![col_ix],
                &samples,
                &Given::Nothing,
                None,
            );

            let h: f64 = logps.iter().map(|logp| -logp * logp.exp()).sum();

            h * q_recip / (n as f64)
        };

        let h_exact = entropy_single(col_ix, &vec![state]);

        (h_exact, h_sobol)
    }

    #[test]
    fn sobol_single_categorical_entropy_vs_exact() {
        let (h_exact, h_sobol) = sobolo_vs_exact_entropy(2, 10_000);
        assert_relative_eq!(h_exact, h_sobol, epsilon = 1E-12);
    }

    #[test]
    fn sobol_single_gaussian_entropy_vs_exact() {
        let (h_exact, h_sobol) = sobolo_vs_exact_entropy(0, 10_000);
        assert_relative_eq!(h_exact, h_sobol, epsilon = 1E-8);
    }
}

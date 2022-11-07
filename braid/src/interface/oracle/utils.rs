use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::f64::{INFINITY, NEG_INFINITY};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;

use braid_cc::feature::{ColModel, FType, Feature};
use braid_cc::state::State;
use braid_data::label::Label;
use braid_data::Datum;
use braid_stats::labeler::Labeler;
use braid_stats::MixtureType;
use braid_utils::{argmax, logsumexp, transpose};
use rv::dist::{Categorical, Gaussian, Mixture, Poisson};
use rv::traits::{Entropy, KlDivergence, Mode, QuadBounds, Rv, Variance};

use crate::interface::Given;
use crate::optimize::{fmin_bounded, fmin_brute};
use crate::{HasStates, OracleT};

use super::error::ColumnMaxiumLogPError;

pub struct ColumnMaximumLogpCache {
    pub(crate) state_ixs_hash: u64,
    pub(crate) col_ixs_hash: u64,
    pub(crate) given_hash: u64,
    pub(crate) cache: Vec<Vec<f64>>,
}

fn hash_with_default<H: Hash>(x: &H) -> u64 {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

impl ColumnMaximumLogpCache {
    pub fn from_oracle<O: OracleT + HasStates>(
        oracle: &O,
        col_ixs: &[usize],
        given: &Given,
        states_ixs_opt: Option<&[usize]>,
    ) -> Self {
        let states = select_states(oracle.states(), states_ixs_opt);
        let state_ixs = state_ixs(oracle.n_states(), states_ixs_opt);

        let cache = states
            .iter()
            .zip(state_ixs.iter())
            .map(|(state, &state_ix)| {
                col_ixs
                    .iter()
                    .map(|&col_ix| {
                        // If this unwrap fails, its our fault -- bad unput validation
                        let ftype = oracle.ftype(col_ix).unwrap();
                        let x = predict(col_ix, ftype, given, &[state]);
                        oracle
                            .logp(
                                &[col_ix],
                                &[vec![x]],
                                &Given::Nothing,
                                Some(&[state_ix]),
                            )
                            .unwrap()[0]
                    })
                    .collect()
            })
            .collect();

        Self {
            state_ixs_hash: hash_with_default(&state_ixs),
            col_ixs_hash: hash_with_default(&col_ixs),
            given_hash: hash_with_default(given),
            cache,
        }
    }

    pub fn validate<O: HasStates>(
        &self,
        oracle: &O,
        col_ixs: &[usize],
        given: &Given,
        states_ixs_opt: Option<&[usize]>,
    ) -> Result<(), ColumnMaxiumLogPError> {
        let state_ixs = state_ixs(oracle.n_states(), states_ixs_opt);

        if hash_with_default(&state_ixs) != self.state_ixs_hash {
            return Err(ColumnMaxiumLogPError::InvalidStateIndices);
        }

        if hash_with_default(&col_ixs) != self.col_ixs_hash {
            return Err(ColumnMaxiumLogPError::InvalidColumnIndices);
        }

        if hash_with_default(given) != self.given_hash {
            return Err(ColumnMaxiumLogPError::InvalidGiven);
        }

        Ok(())
    }
}

pub(crate) fn select_states<'s>(
    states: &'s [State],
    states_ixs_opt: Option<&[usize]>,
) -> Vec<&'s State> {
    match states_ixs_opt {
        Some(state_ixs) => state_ixs.iter().map(|&ix| &states[ix]).collect(),
        None => states.iter().collect(),
    }
}

pub(crate) fn state_ixs(
    n_states: usize,
    states_ixs_opt: Option<&[usize]>,
) -> Vec<usize> {
    match states_ixs_opt {
        Some(state_ixs) => state_ixs.to_vec(),
        None => (0..n_states).collect(),
    }
}

/// Generates samples
pub struct Simulator<'s, R: rand::Rng> {
    rng: &'s mut R,
    /// A list of the states
    states: &'s [&'s State],
    /// The view weights for each state
    weights: &'s [BTreeMap<usize, Vec<f64>>],
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
        states: &'s [&'s State],
        weights: &'s [BTreeMap<usize, Vec<f64>>],
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
pub struct Calculator<'s, Xs>
where
    Xs: Iterator,
    Xs::Item: Borrow<Vec<Datum>>,
{
    /// A list of the states
    states: &'s [&'s State],
    /// The view weights for each state
    weights: &'s [BTreeMap<usize, Vec<f64>>],
    /// List of state indices from which to simulate
    col_ixs: &'s [usize],
    values: &'s mut Xs,
    /// Holds the values of logp under each state. Prevents reallocations of
    /// vectors for every logp computation.
    state_logps: Vec<f64>,
    col_max_logps: Option<&'s [Vec<f64>]>,
}

impl<'s, Xs> Calculator<'s, Xs>
where
    Xs: Iterator,
    Xs::Item: Borrow<Vec<Datum>>,
{
    pub fn new(
        values: &'s mut Xs,
        states: &'s [&'s State],
        weights: &'s [BTreeMap<usize, Vec<f64>>],
        col_ixs: &'s [usize],
    ) -> Self {
        Self {
            values,
            weights,
            states,
            col_ixs,
            state_logps: vec![0.0; states.len()],
            col_max_logps: None,
        }
    }

    pub fn new_scaled(
        values: &'s mut Xs,
        states: &'s [&'s State],
        weights: &'s [BTreeMap<usize, Vec<f64>>],
        col_ixs: &'s [usize],
        col_max_logps: &'s [Vec<f64>],
    ) -> Self {
        Self {
            values,
            weights,
            states,
            col_ixs,
            state_logps: vec![0.0; states.len()],
            col_max_logps: Some(col_max_logps),
        }
    }
}

impl<'s, Xs> Iterator for Calculator<'s, Xs>
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
                self.states
                    .iter()
                    .zip(self.weights.iter())
                    .enumerate()
                    .for_each(|(i, (state, weights))| {
                        let logp = single_val_logp(
                            state,
                            col_ixs,
                            xs.borrow(),
                            weights.clone(),
                            self.col_max_logps.map(|cmlp| cmlp[i].as_slice()),
                        );
                        self.state_logps[i] = logp;
                    });
                let logp = logsumexp(&self.state_logps) - ln_n;
                if self.col_max_logps.is_some() {
                    // Geometric mean
                    Some(logp / self.col_ixs.len() as f64)
                } else {
                    Some(logp)
                }
            }
            None => None,
        }
    }
}

pub fn load_states<P: AsRef<Path>>(filenames: Vec<P>) -> Vec<State> {
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
#[inline]
pub fn given_weights(
    states: &[&State],
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    states
        .iter()
        .map(|state| single_state_weights(state, col_ixs, given))
        .collect()
}

#[inline]
pub fn given_exp_weights(
    states: &[&State],
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    states
        .iter()
        .map(|state| single_state_exp_weights(state, col_ixs, given))
        .collect()
}

#[inline]
pub fn state_weights(
    states: &[&State],
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    states
        .iter()
        .map(|state| single_state_weights(state, col_ixs, given))
        .collect()
}

#[inline]
pub fn state_exp_weights(
    states: &[State],
    col_ixs: &[usize],
    given: &Given,
) -> Vec<BTreeMap<usize, Vec<f64>>> {
    states
        .iter()
        .map(|state| single_state_exp_weights(state, col_ixs, given))
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
            view_weights
                .entry(view_ix)
                .or_insert_with(|| single_view_weights(state, view_ix, given));
        });

    view_weights
}

#[inline]
pub fn single_state_exp_weights(
    state: &State,
    col_ixs: &[usize],
    given: &Given,
) -> BTreeMap<usize, Vec<f64>> {
    let mut view_weights: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    col_ixs
        .iter()
        .map(|&ix| state.asgn.asgn[ix])
        .for_each(|view_ix| {
            view_weights.entry(view_ix).or_insert_with(|| {
                single_view_exp_weights(state, view_ix, given)
            });
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
                    view.ftrs[&id].accum_weights(datum, &mut weights, None);
                }
            }
            let z = logsumexp(&weights);
            weights.iter_mut().for_each(|w| *w -= z);
        }
        Given::Nothing => (),
    }
    weights
}

#[inline]
fn single_view_exp_weights(
    state: &State,
    target_view_ix: usize,
    given: &Given,
) -> Vec<f64> {
    let view = &state.views[target_view_ix];
    let mut weights = view.weights.clone();

    match given {
        Given::Conditions(ref conditions) => {
            conditions.iter().for_each(|(ix, datum)| {
                let in_target_view = state.asgn.asgn[*ix] == target_view_ix;
                if in_target_view {
                    view.ftrs[ix].accum_exp_weights(datum, &mut weights);
                }
            });
            let z = weights.iter().sum::<f64>();
            weights.iter_mut().for_each(|w| *w /= z);
        }
        Given::Nothing => (),
    }
    weights
}

// Probability calculation
// -----------------------

/// Compute the probability of values under the state
///
/// # Notes
///
/// The mixture likelihood is
///
///  f(x) = Σ πᵢ f(x | θᵢ)
///
/// The scaled likelihood is
///
///  f(x) = Σ πᵢ f(x | θᵢ) / f(mode(θᵢ))
///
/// # Arguments
///
/// - state: The state
/// - col_ixs: The column indices that each entry in each vector in `vals`
///   comes from
/// - vals: A vector of value rows. `vals[i][j]` is a datum from the column
///   with index `col_ixs[j]`. The function returns a vector with an entry for
///   each row in `vals`.
/// - given: An optional set of conditions on the targets for p(vals | given).
/// - view_weights_opt: Optional precomputed weights.
/// - col_max_logps: If supplied, the logp component contributed by each column
///   will be normalized to [0, 1]. `col_max_logps[i]` should be the max log
///   likelihood of column `col_ixs[i]` given the `Given`.
pub fn state_logp(
    state: &State,
    col_ixs: &[usize],
    vals: &[Vec<Datum>],
    given: &Given,
    view_weights_opt: Option<&BTreeMap<usize, Vec<f64>>>,
    col_max_logps: Option<&[f64]>,
) -> Vec<f64> {
    match view_weights_opt {
        Some(view_weights) => vals
            .iter()
            .map(|val| {
                single_val_logp(
                    state,
                    col_ixs,
                    val,
                    view_weights.clone(),
                    col_max_logps,
                )
            })
            .collect(),
        None => {
            let mut view_weights = single_state_weights(state, col_ixs, given);

            // normalize view weights
            for weights in view_weights.values_mut() {
                let logz = logsumexp(weights);
                weights.iter_mut().for_each(|w| *w -= logz);
            }
            vals.iter()
                .map(|val| {
                    single_val_logp(
                        state,
                        col_ixs,
                        val,
                        view_weights.clone(),
                        col_max_logps,
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
    col_max_logps: Option<&[f64]>,
) -> f64 {
    // TODO: is there a way to do this without cloning the view_weights?
    col_ixs
        .iter()
        .zip(val)
        .map(|(col_ix, datum)| (col_ix, state.asgn.asgn[*col_ix], datum))
        .enumerate()
        .for_each(|(i, (col_ix, view_ix, datum))| {
            state.views[view_ix].ftrs[col_ix].accum_weights(
                datum,
                view_weights.get_mut(&view_ix).unwrap(),
                col_max_logps.map(|inner| inner[i]),
            );
        });

    view_weights.values().map(|logps| logsumexp(logps)).sum()
}
pub fn state_likelihood(
    state: &State,
    col_ixs: &[usize],
    vals: &[Vec<Datum>],
    given: &Given,
    view_exp_weights_opt: Option<&BTreeMap<usize, Vec<f64>>>,
) -> Vec<f64> {
    match view_exp_weights_opt {
        Some(view_exp_weights) => vals
            .iter()
            .map(|val| {
                single_val_likelihood(state, col_ixs, val, view_exp_weights)
            })
            .collect(),
        None => {
            let mut view_exp_weights =
                single_state_exp_weights(state, col_ixs, given);

            // normalize view weights
            for weights in view_exp_weights.values_mut() {
                let z = weights.iter().sum::<f64>();
                weights.iter_mut().for_each(|w| *w /= z);
            }

            vals.iter()
                .map(|val| {
                    single_val_likelihood(
                        state,
                        col_ixs,
                        val,
                        &view_exp_weights,
                    )
                })
                .collect()
        }
    }
}

fn single_val_likelihood(
    state: &State,
    col_ixs: &[usize],
    val: &[Datum],
    view_exp_weights: &BTreeMap<usize, Vec<f64>>,
) -> f64 {
    view_exp_weights
        .iter()
        .fold(1.0, |prod, (&view_ix, weights)| {
            let view = &state.views[view_ix];
            // lookup for column indices and data assigned to the view
            let view_data: Vec<(usize, Datum)> = col_ixs
                .iter()
                .zip(val.iter())
                .filter(|(ix, _)| view.ftrs.contains_key(ix))
                .map(|(ix, val)| (*ix, val.clone()))
                .collect();

            prod * weights
                .iter()
                .enumerate()
                .map(|(k, &w)| {
                    view_data.iter().fold(w, |acc, (col_ix, val)| {
                        acc * view.ftrs[col_ix].cpnt_likelihood(val, k)
                    })
                })
                .sum::<f64>()
        })
}

// Imputation
// ----------
fn impute_bounds(states: &[&State], col_ix: usize) -> (f64, f64) {
    states
        .iter()
        .map(|state| state.impute_bounds(col_ix).unwrap())
        .fold((INFINITY, NEG_INFINITY), |(min, max), (lower, upper)| {
            (min.min(lower), max.max(upper))
        })
}

pub fn continuous_impute(
    states: &[&State],
    row_ix: usize,
    col_ix: usize,
) -> f64 {
    let cpnts: Vec<Gaussian> = states
        .iter()
        .map(|state| {
            state
                .component(row_ix, col_ix)
                .try_into()
                .expect("Unexpected column type")
        })
        .collect();

    if cpnts.len() == 1 {
        cpnts[0].mu()
    } else {
        let f = |x: f64| {
            let logfs: Vec<f64> =
                cpnts.iter().map(|cpnt| cpnt.ln_f(&x)).collect();
            -logsumexp(&logfs)
        };

        let bounds = impute_bounds(states, col_ix);
        let n_grid = 100;
        let step_size = (bounds.1 - bounds.0) / (n_grid as f64);
        let x0 = fmin_brute(&f, bounds, n_grid);
        fmin_bounded(f, (x0 - step_size, x0 + step_size), None, None)
    }
}

pub fn categorical_impute(
    states: &[&State],
    row_ix: usize,
    col_ix: usize,
) -> u8 {
    let cpnts: Vec<Categorical> = states
        .iter()
        .map(|state| {
            state
                .component(row_ix, col_ix)
                .try_into()
                .expect("Unexpected column type")
        })
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

pub fn labeler_impute(
    states: &[&State],
    row_ix: usize,
    col_ix: usize,
) -> Label {
    let cpnts: Vec<Labeler> = states
        .iter()
        .map(|state| {
            state
                .component(row_ix, col_ix)
                .try_into()
                .expect("Unexpected column type")
        })
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

pub fn count_impute(states: &[&State], row_ix: usize, col_ix: usize) -> u32 {
    use braid_utils::MinMax;
    use rv::traits::Mean;

    let cpnts: Vec<Poisson> = states
        .iter()
        .map(|state| {
            state
                .component(row_ix, col_ix)
                .try_into()
                .expect("Unexpected column type")
        })
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

pub fn entropy_single(col_ix: usize, states: &[State]) -> f64 {
    let mixtures = states
        .iter()
        .map(|state| state.feature_as_mixture(col_ix))
        .collect();
    let mixture = MixtureType::combine(mixtures);
    mixture.entropy()
}

fn sort_mixture_by_mode<Fx>(mm: Mixture<Fx>) -> Mixture<Fx>
where
    Fx: Mode<f64>,
{
    let mut components: Vec<(f64, Fx)> = mm.into();
    components.sort_by(|a, b| {
        a.1.mode()
            .partial_cmp(&b.1.mode())
            .unwrap_or(std::cmp::Ordering::Less)
    });
    Mixture::<Fx>::try_from(components).unwrap()
}

fn continuous_mixture_quad_points<Fx>(mm: &Mixture<Fx>) -> Vec<f64>
where
    Fx: Mode<f64> + Variance<f64>,
{
    let mut state: (Option<f64>, Option<f64>) = (None, None);
    let m = 2.0;
    mm.components()
        .iter()
        .filter_map(|cpnt| {
            let mode = cpnt.mode();
            let std = cpnt.variance().map(|v| v.sqrt());
            match (&state, (mode, std)) {
                ((Some(m1), s1), (Some(m2), s2)) => {
                    if (m2 - *m1)
                        > (m * s1.unwrap_or(INFINITY))
                            .min(m * s2.unwrap_or(INFINITY))
                    {
                        state = (mode, std);
                        Some(m2)
                    } else {
                        None
                    }
                }
                ((None, _), (Some(m2), _)) => {
                    state = (mode, std);
                    Some(m2)
                }
                _ => None,
            }
        })
        .collect()
}

macro_rules! dep_ind_col_mixtures {
    ($states: ident, $col_a: ident, $col_b: ident, $fx: ident) => {{
        // Mixtures of col_a for which col_a and col_b are in the same view
        // (dependent).
        let mut mms_dep = Vec::new();
        // Mixtures of col_a for which col_a and col_b are in different views
        // (independent).
        let mut mms_ind = Vec::new();
        // The proportion of times the columns are in the same view (same as
        // dependence probability).
        let mut weight = 0.0;
        $states.iter().for_each(|state| {
            let mm = match state.feature_as_mixture($col_a) {
                MixtureType::$fx(mm) => mm,
                _ => panic!("Unexpected MixtureType"),
            };

            if state.asgn.asgn[$col_a] == state.asgn.asgn[$col_b] {
                weight += 1.0;
                mms_dep.push(mm);
            } else {
                mms_ind.push(mm);
            }
        });

        weight /= $states.len() as f64;

        // Combine the mixtures within each type into one big mixture for each
        // type.
        (weight, Mixture::combine(mms_dep), Mixture::combine(mms_ind))
    }};
}

/// Joint entropy H(X, Y) where X is Categorical and Y is Gaussian
pub fn categorical_gaussian_entropy_dual(
    col_cat: usize,
    col_gauss: usize,
    states: &[State],
) -> f64 {
    use rv::misc::{gauss_legendre_quadrature_cached, gauss_legendre_table};
    use std::cell::RefCell;
    use std::collections::HashMap;

    // get a mixture model of the Gaussian component to compute the quad points
    let (dep_weight, gm_dep, gm_ind) =
        dep_ind_col_mixtures!(states, col_gauss, col_cat, Gaussian);
    let (_, cm_dep, cm_ind) =
        dep_ind_col_mixtures!(states, col_cat, col_gauss, Categorical);

    // Get the number of values the categorical column support. Can never exceed
    // u8::MAX (255).
    let cat_k = match states[0].feature(col_cat) {
        ColModel::Categorical(cm) => u8::try_from(cm.prior.k())
            .expect("Categorical k exceeded u8 max value"),
        _ => panic!("Expected ColModel::Categorical"),
    };

    // Divide the function into nicely behaved intervals
    let (points, lower, upper) = {
        let gmm = Mixture::combine(vec![gm_ind.clone(), gm_dep.clone()]);
        let gmm = sort_mixture_by_mode(gmm);
        let points = continuous_mixture_quad_points(&gmm);
        let (lower, upper) = gmm.quad_bounds();
        (points, lower, upper)
    };

    // Make sure the dependent state mixtures line up
    assert_eq!(cm_dep.k(), gm_dep.k());
    cm_dep
        .weights()
        .iter()
        .zip(gm_dep.weights().iter())
        .for_each(|(wc, wg)| assert!((wc - wg).abs() < 1e-12));

    // If the columns are either always in the same view or never in the same
    // view across states, we may run into empty container errors, so we keep
    // track here so we don't compute things we don't need to and potentially
    // pass empty containers where they're not expected.
    let has_dep_states = gm_dep.k() > 0;
    let has_ind_states = gm_ind.k() > 0;

    // order of the polynomial for gauss-legendre quadrature
    let quad_level = 16;

    // Super aggressive caching. You can't hash a f64, so we create a structure
    // to transmute it to a u64 so we can use it as an index in our cache
    #[derive(Hash, Clone, Copy, PartialEq, Eq)]
    struct F64(u64);

    impl F64 {
        fn new(x: f64) -> Self {
            // The quadrature points should be exactly the same each time. If
            // that doesn't turn out the be the case, we can round x to like 14
            // decimals or something.
            Self(x.to_bits())
        }
    }

    let ind_cache: RefCell<HashMap<F64, f64>> = RefCell::new(HashMap::new());
    let dep_cache: RefCell<HashMap<F64, Vec<f64>>> =
        RefCell::new(HashMap::new());

    // Pre-generate the weights and roots for the quadrate since it never
    // changes and requires allocating a couple of vecs each time the quadrature
    // is run.
    let gl_cache = gauss_legendre_table(quad_level);

    // NOTE: this will take a really long time when k is large
    -(0..cat_k)
        .map(|k| {
            // NOTE: I've chose to use the logp instead of vanilla 'p'. It
            // doesn't really change the runtime.
            let ind_cat_f = if has_ind_states {
                cm_ind.ln_f(&k)
            } else {
                // Note, it shouldn't matter what we return here because the
                // weight for the independent mixture will be 0
                1.0 // ln(0)
            };

            let dep_cat_fs: Vec<f64> = cm_dep
                .weights()
                .iter()
                .zip(cm_dep.components().iter())
                .map(|(w, cpnt)| w.ln() + cpnt.ln_f(&k))
                .collect();

            let quad_fn = |y: f64| {
                // We have to compute things differently for states in which the
                // two columns are dependent and independent. The dependednt
                // computation is a bit more complicated.
                let dep_cpnt = if has_dep_states {
                    let mut m = dep_cache.borrow_mut();
                    let ln_fys = m.entry(F64::new(y)).or_insert_with(|| {
                        gm_dep
                            .components()
                            .iter()
                            .map(|cpnt| cpnt.ln_f(&y))
                            .collect()
                    });
                    // This does manually what state_logp does, but this is
                    // faster because it's less general
                    let cpnts: Vec<f64> = dep_cat_fs
                        .iter()
                        .zip(ln_fys)
                        .map(|(w, g)| w + *g)
                        .collect();
                    logsumexp(&cpnts)
                } else {
                    // Note, it shouldn't matter what we return here because the
                    // weight for the dependent mixture will be 0
                    1.0 // ln(0)
                };

                // We can basically cache the entire independent computation, so
                // things will be faster the fewer states that have the columns
                // in the same view
                let ind_cpnt = if has_ind_states {
                    let mut m = ind_cache.borrow_mut();
                    let ln_fy =
                        m.entry(F64::new(y)).or_insert_with(|| gm_ind.ln_f(&y));
                    ind_cat_f + *ln_fy
                } else {
                    assert_eq!(dep_weight, 1.0);
                    0.0
                };

                // add the weighted sums of the independent-columns mixture and
                // the dependent-columns mixture
                let ln_f = logsumexp(&[
                    dep_weight.ln() + dep_cpnt,
                    (1.0 - dep_weight).ln() + ind_cpnt,
                ]);

                ln_f * ln_f.exp()
            };

            let last_ix = points.len() - 1;

            let q_a = gauss_legendre_quadrature_cached(
                quad_fn,
                (lower, points[0]),
                &gl_cache.0,
                &gl_cache.1,
            );

            let q_b = gauss_legendre_quadrature_cached(
                quad_fn,
                (points[last_ix], upper),
                &gl_cache.0,
                &gl_cache.1,
            );

            let q_m = if points.len() == 1 {
                0.0
            } else {
                let mut left = points[0];
                points
                    .iter()
                    .skip(1)
                    .map(|&x| {
                        let q = gauss_legendre_quadrature_cached(
                            quad_fn,
                            (left, x),
                            &gl_cache.0,
                            &gl_cache.1,
                        );

                        left = x;
                        q
                    })
                    .sum::<f64>()
            };

            q_a + q_m + q_b
        })
        .sum::<f64>()
}

/// Computes entropy among categorical columns exactly via enumeration
pub fn categorical_joint_entropy(col_ixs: &[usize], states: &[State]) -> f64 {
    let ranges = col_ixs
        .iter()
        .map(|&ix| {
            let cpnt: Categorical = states[0]
                .component(0, ix)
                .try_into()
                .expect("Unexpected column type");
            cpnt.k() as u8
        })
        .collect();

    let vals: Vec<_> = braid_utils::CategoricalCartProd::new(ranges)
        .map(|mut xs| {
            let vals: Vec<_> = xs.drain(..).map(Datum::Categorical).collect();
            vals
        })
        .collect();

    // TODO: this is a pattern that appears a lot. I should DRY it.
    let logps: Vec<Vec<f64>> = states
        .iter()
        .map(|state| {
            state_logp(state, col_ixs, &vals, &Given::Nothing, None, None)
        })
        .collect();

    let ln_n_states = (states.len() as f64).ln();

    transpose(&logps)
        .iter()
        .map(|lps| logsumexp(lps) - ln_n_states)
        .fold(0.0, |acc, lp| lp.mul_add(-lp.exp(), acc))
}

/// Joint entropy H(X, Y) where both X and Y are Categorical
pub fn categorical_entropy_dual(
    col_a: usize,
    col_b: usize,
    states: &[State],
) -> f64 {
    // TODO: We could probably do a lot of pre-computation and caching like we
    // do in categorical_gaussian_entropy_dual, but this function is really fast
    // as it is, so it's probably not a good candidate for optimization
    if col_a == col_b {
        return entropy_single(col_a, states);
    }

    let k_a = match states[0].feature(col_a) {
        ColModel::Categorical(cm) => cm.prior.k(),
        _ => panic!("Expected ColModel::Categorical"),
    };

    let k_b = match states[0].feature(col_b) {
        ColModel::Categorical(cm) => cm.prior.k(),
        _ => panic!("Expected ColModel::Categorical"),
    };

    let mut vals: Vec<Vec<Datum>> = Vec::with_capacity(k_a * k_b);
    for i in 0..k_a {
        for j in 0..k_b {
            vals.push(vec![
                Datum::Categorical(i as u8),
                Datum::Categorical(j as u8),
            ]);
        }
    }

    let view_weights =
        state_exp_weights(states, &[col_a, col_b], &Given::Nothing);

    let ps = {
        let mut ps = vec![0_f64; vals.len()];
        states
            .iter()
            .zip(view_weights.iter())
            .for_each(|(state, weights)| {
                state_likelihood(
                    state,
                    &[col_a, col_b],
                    &vals,
                    &Given::Nothing,
                    Some(weights),
                )
                .drain(..)
                .enumerate()
                .for_each(|(ix, p)| {
                    ps[ix] += p;
                });
            });

        let sf = states.len() as f64;
        ps.iter_mut().for_each(|p| *p /= sf);
        ps
    };

    ps.iter().map(|p| -p * p.ln()).sum::<f64>()
}

// Finds the first x such that
fn count_pr_limit(col: usize, mass: f64, states: &[State]) -> (u32, u32) {
    use rv::traits::{Cdf, Mean};

    let lower_threshold = (1.0 - mass) / 2.0;
    let upper_threshold = mass - (1.0 - mass) / 2.0;

    let mixtures = states
        .iter()
        .map(|state| {
            if let MixtureType::Poisson(mm) = state.feature_as_mixture(col) {
                mm
            } else {
                panic!("expected count type for column {}", col);
            }
        })
        .collect::<Vec<_>>();

    let mm = Mixture::combine(mixtures);
    let max_mean = mm
        .components()
        .iter()
        .map(|cpnt| {
            let mean: u32 = cpnt.mean().unwrap().round() as u32;
            mean
        })
        .max()
        .unwrap();

    let lower = (0_u32..)
        .find_map(|x| {
            if mm.cdf(&x) > lower_threshold {
                // make sure the lower bound is >= 0
                Some(x.saturating_sub(1))
            } else {
                None
            }
        })
        .unwrap();

    #[allow(clippy::unnecessary_find_map)]
    let upper = (max_mean..)
        .find_map(|x| {
            if mm.cdf(&x) > upper_threshold {
                Some(x)
            } else {
                None
            }
        })
        .unwrap();

    assert!(lower < upper);

    (lower, upper)
}

/// Joint entropy H(X, Y) where both X and Y are Categorical
pub fn count_entropy_dual(col_a: usize, col_b: usize, states: &[State]) -> f64 {
    if col_a == col_b {
        return entropy_single(col_a, states);
    }

    let mass: f64 = 1_f64 - 1E-16;
    let (a_lower, a_upper) = count_pr_limit(col_a, mass, states);
    let (b_lower, b_upper) = count_pr_limit(col_b, mass, states);

    let nx = (a_upper - a_lower) * (b_upper - b_lower);
    let mut vals: Vec<Vec<Datum>> = Vec::with_capacity(nx as usize);

    // TODO: make this into an iterator
    for a in a_lower..a_upper {
        for b in b_lower..b_upper {
            vals.push(vec![Datum::Count(a), Datum::Count(b)]);
        }
    }

    let logps: Vec<Vec<f64>> = states
        .iter()
        .map(|state| {
            state_logp(
                state,
                &[col_a, col_b],
                &vals,
                &Given::Nothing,
                None,
                None,
            )
        })
        .collect();

    let ln_n_states = (states.len() as f64).ln();

    transpose(&logps)
        .iter()
        .map(|lps| logsumexp(lps) - ln_n_states)
        .fold(0.0, |acc, lp| lp.mul_add(-lp.exp(), acc))
}

// Prediction
// ----------
pub(crate) fn predict(
    col_ix: usize,
    ftype: FType,
    given: &Given,
    states: &[&State],
) -> Datum {
    match ftype {
        FType::Continuous => {
            let x = continuous_predict(states, col_ix, given);
            Datum::Continuous(x)
        }
        FType::Categorical => {
            let x = categorical_predict(states, col_ix, given);
            Datum::Categorical(x)
        }
        FType::Labeler => {
            let x = labeler_predict(states, col_ix, given);
            Datum::Label(x)
        }
        FType::Count => {
            let x = count_predict(states, col_ix, given);
            Datum::Count(x)
        }
    }
}

pub fn continuous_predict(
    states: &[&State],
    col_ix: usize,
    given: &Given,
) -> f64 {
    let mm = {
        let mixtures = states
            .iter()
            .map(|state| {
                let view_ix = state.asgn.asgn[col_ix];
                // NOTE: There is a slight speedup from using given_exp_weights,
                // but at the cost of panics when there is a large number of
                // conditions in the given: underflow causes all the weights to
                // be zero, which causes a constructor error in Mixture::new
                let weights = &given_weights(&[state], &[col_ix], given)[0];
                let mut mm_weights: Vec<f64> = state.views[view_ix]
                    .weights
                    .iter()
                    .zip(weights[&view_ix].iter())
                    .map(|(&w1, &w2)| w1 + w2)
                    .collect();

                let z: f64 = logsumexp(&mm_weights);
                mm_weights.iter_mut().for_each(|w| *w = (*w - z).exp());

                match state.views[view_ix].ftrs[&col_ix].to_mixture(mm_weights)
                {
                    MixtureType::Gaussian(m) => m,
                    _ => panic!("invalid MixtureType for continuous predict"),
                }
            })
            .collect();

        let mm = Mixture::combine(mixtures);

        // sorts the mixture components in ascending order by their means/modes
        sort_mixture_by_mode(mm)
    };

    let f = |x: f64| -mm.f(&x);

    // We find the mode in the mixture model with the highest likelihood then
    // build everything around that mode
    let eval_points = continuous_mixture_quad_points(&mm);
    let n_eval_points = eval_points.len();

    if n_eval_points == 1 {
        return eval_points[0];
    }

    let min_ix = eval_points
        .iter()
        .enumerate()
        .map(|(ix, &x)| (ix, f(x)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;

    // Check whether the first or last modes are the highest likelihood
    let (ix_left, ix_right) = if min_ix == 0 {
        (0, 1)
    } else if min_ix == n_eval_points - 1 {
        (n_eval_points - 2, n_eval_points - 1)
    } else {
        (min_ix - 1, min_ix + 1)
    };

    let left = eval_points[ix_left];
    let right = eval_points[ix_right];
    let n_steps = 20;
    let step_size = (right - left) / n_steps as f64;

    // Use a grid search to narrow down the range
    let x0 = fmin_brute(&f, (left, right), n_steps);
    fmin_bounded(f, (x0 - step_size, x0 + step_size), None, None)
}

pub fn categorical_predict(
    states: &[&State],
    col_ix: usize,
    given: &Given,
) -> u8 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let state_weights = state_weights(states, &col_ixs, given);

    let f = |x: u8| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Categorical(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, given, Some(view_weights), None)
                    [0]
            })
            .collect();
        logsumexp(&scores)
    };

    let k: u8 = match states[0].feature(col_ix) {
        ColModel::Categorical(ftr) => ftr.prior.k() as u8,
        _ => panic!("FType mitmatch."),
    };

    let fs: Vec<f64> = (0..k).map(f).collect();
    argmax(&fs) as u8
}

// XXX: Not 100% sure how to predict `label` given `truth'. For now, we're
// going to predict (label, truth), given other columns.
pub fn labeler_predict(
    states: &[&State],
    col_ix: usize,
    given: &Given,
) -> Label {
    let col_ixs: Vec<usize> = vec![col_ix];

    let state_weights = state_weights(states, &col_ixs, given);

    let f = |x: Label| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Label(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, given, Some(view_weights), None)
                    [0]
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

pub fn count_predict(states: &[&State], col_ix: usize, given: &Given) -> u32 {
    let col_ixs: Vec<usize> = vec![col_ix];

    let state_weights = state_weights(states, &col_ixs, given);

    let ln_fx = |x: u32| {
        let y: Vec<Vec<Datum>> = vec![vec![Datum::Count(x)]];
        let scores: Vec<f64> = states
            .iter()
            .zip(state_weights.iter())
            .map(|(state, view_weights)| {
                state_logp(state, &col_ixs, &y, given, Some(view_weights), None)
                    [0]
            })
            .collect();
        logsumexp(&scores)
    };

    let (lower, upper) = {
        let (lower, upper) = impute_bounds(states, col_ix);
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

fn jsd_mixture<Fx>(mut components: Vec<Mixture<Fx>>) -> f64
where
    MixtureType: From<Mixture<Fx>>,
{
    // TODO: we could do all this with the usual Rv Mixture functions if it
    // wasn't for that damned Labeler type
    let n_states = components.len() as f64;
    let mut h_cpnts = 0_f64;
    let mts: Vec<MixtureType> = components
        .drain(..)
        .map(|mm| {
            let mt = MixtureType::from(mm);
            // h_cpnts += mt.entropy();
            h_cpnts += mt.entropy();
            mt
        })
        .collect();

    // let mt: MixtureType = mm.into();
    let mm = MixtureType::combine(mts);
    let h_mixture = mm.entropy();

    h_mixture - h_cpnts / n_states
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

        jsd_mixture(mix_models)
    }};
}

pub fn predict_uncertainty(
    states: &[State],
    col_ix: usize,
    given: &Given,
    states_ixs_opt: Option<&[usize]>,
) -> f64 {
    let ftype = {
        let view_ix = states[0].asgn.asgn[col_ix];
        states[0].views[view_ix].ftrs[&col_ix].ftype()
    };
    let states = select_states(states, states_ixs_opt);
    match ftype {
        FType::Continuous => predunc_arm!(states, col_ix, given, Gaussian),
        FType::Categorical => predunc_arm!(states, col_ix, given, Categorical),
        FType::Labeler => predunc_arm!(states, col_ix, given, Labeler),
        FType::Count => predunc_arm!(states, col_ix, given, Poisson),
    }
}

macro_rules! js_impunc_arm {
    ($k: expr, $row_ix: expr, $states: expr, $ftr: expr, $variant: ident) => {{
        let n_states = $states.len();
        let col_ix = $ftr.id;
        let mut cpnts = Vec::with_capacity(n_states);
        cpnts.push($ftr.components[$k].fx.clone());
        for i in 1..n_states {
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

pub fn js_impute_uncertainty(
    states: &[State],
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

pub fn kl_impute_uncertainty(
    states: &[State],
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

    let n_states = states.len() as f64;
    kl_sum / n_states.mul_add(n_states, -n_states)
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
        states: &[State],
    ) -> f64 {
        let cpnt: Categorical =
            states[0].component(0, col_ix).try_into().unwrap();
        let k = cpnt.k();

        let mut vals: Vec<Vec<Datum>> = Vec::with_capacity(k);
        for i in 0..k {
            vals.push(vec![Datum::Categorical(i as u8)]);
        }

        let logps: Vec<Vec<f64>> = states
            .iter()
            .map(|state| {
                state_logp(state, &[col_ix], &vals, &Given::Nothing, None, None)
            })
            .collect();

        let ln_n_states = (states.len() as f64).ln();

        transpose(&logps)
            .iter()
            .map(|lps| logsumexp(lps) - ln_n_states)
            .fold(0.0, |acc, lp| lp.mul_add(-lp.exp(), acc))
    }

    #[test]
    fn single_continuous_column_weights_no_given() {
        let state = get_single_continuous_state_from_yaml();

        let weights = single_view_weights(&state, 0, &Given::Nothing);

        assert_relative_eq!(weights[0], -std::f64::consts::LN_2, epsilon = TOL);
        assert_relative_eq!(weights[1], -std::f64::consts::LN_2, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_weights_given() {
        let state = get_single_continuous_state_from_yaml();
        let given = Given::Conditions(vec![(0, Datum::Continuous(0.5))]);

        let weights = single_view_weights(&state, 0, &given);
        let target = {
            let mut unnormed_targets =
                vec![-2.857_054_917_013_031_5, -16.598_938_533_204_67];
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
        let states: Vec<&State> = states.iter().collect();

        let x = continuous_predict(&states, 0, &Given::Nothing);
        assert_relative_eq!(x, -0.12, epsilon = 1E-5);
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let states = get_states_from_yaml();

        let weights_0 = single_view_weights(&states[0], 0, &Given::Nothing);

        assert_relative_eq!(
            weights_0[0],
            -std::f64::consts::LN_2,
            epsilon = TOL
        );
        assert_relative_eq!(
            weights_0[1],
            -std::f64::consts::LN_2,
            epsilon = TOL
        );

        let weights_1 = single_view_weights(&states[0], 1, &Given::Nothing);

        assert_relative_eq!(
            weights_1[0],
            -1.386_294_361_119_890_6,
            epsilon = TOL
        );
        assert_relative_eq!(
            weights_1[1],
            -0.287_682_072_451_780_9,
            epsilon = TOL
        );
    }

    #[test]
    fn single_view_weights_vs_exp() {
        let states = get_states_from_yaml();
        let weights_0 = single_view_weights(&states[0], 0, &Given::Nothing);
        let weights_1 = single_view_weights(&states[0], 1, &Given::Nothing);
        let exp_weights_0 =
            single_view_exp_weights(&states[0], 0, &Given::Nothing);
        let exp_weights_1 =
            single_view_exp_weights(&states[0], 1, &Given::Nothing);

        weights_0
            .iter()
            .zip(exp_weights_0.iter())
            .for_each(|(&w, &e)| assert_relative_eq!(w, e.ln(), epsilon = TOL));

        weights_1
            .iter()
            .zip(exp_weights_1.iter())
            .for_each(|(&w, &e)| assert_relative_eq!(w, e.ln(), epsilon = TOL));
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
                vec![-3.158_958_368_120_129, -1.926_578_447_516_985];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights_0[0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights_0[1], targets[1], epsilon = TOL);
        }

        {
            let unnormed_targets =
                vec![-4.095_863_302_766_923, -0.417_781_136_933_142_9];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights_1[0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights_1[1], targets[1], epsilon = TOL);
        }
    }

    #[test]
    fn single_view_weights_vs_exp_one_given() {
        let given = Given::Conditions(vec![
            (0, Datum::Continuous(0.0)),
            (1, Datum::Continuous(-1.0)),
        ]);

        let states = get_states_from_yaml();
        let weights_0 = single_view_weights(&states[0], 0, &given);
        let weights_1 = single_view_weights(&states[0], 1, &given);
        let exp_weights_0 = single_view_exp_weights(&states[0], 0, &given);
        let exp_weights_1 = single_view_exp_weights(&states[0], 1, &given);

        weights_0
            .iter()
            .zip(exp_weights_0.iter())
            .for_each(|(&w, &e)| assert_relative_eq!(w, e.ln(), epsilon = TOL));

        weights_1
            .iter()
            .zip(exp_weights_1.iter())
            .for_each(|(&w, &e)| assert_relative_eq!(w, e.ln(), epsilon = TOL));
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
                vec![-5.669_175_767_690_254, -9.304_554_786_193_446];
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
                vec![-5.669_175_767_690_254, -9.304_554_786_193_446];
            let z = logsumexp(&unnormed_targets);
            let targets: Vec<_> =
                unnormed_targets.iter().map(|&w| w - z).collect();
            assert_relative_eq!(weights[&0][0], targets[0], epsilon = TOL);
            assert_relative_eq!(weights[&0][1], targets[1], epsilon = TOL);
        }

        {
            let unnormed_targets =
                vec![-4.095_863_302_766_923, -0.417_781_136_933_142_9];
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
            states.iter().collect::<Vec<_>>().as_slice(),
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

    macro_rules! state_logp_vs_exp {
        ($precomp: expr, $state: expr, $col_ixs: expr, $vals: expr, $given: expr) => {{
            let state_weights = single_state_weights($state, $col_ixs, $given);
            let logp = state_logp(
                $state,
                $col_ixs,
                $vals,
                $given,
                if $precomp { Some(&state_weights) } else { None },
                None,
            );

            let state_exp_weights =
                single_state_exp_weights($state, $col_ixs, $given);
            let likeihood = state_likelihood(
                $state,
                $col_ixs,
                $vals,
                $given,
                if $precomp {
                    Some(&state_exp_weights)
                } else {
                    None
                },
            );

            for (&ln_f, &f) in logp.iter().zip(likeihood.iter()) {
                assert_relative_eq!(ln_f, f.ln(), epsilon = TOL)
            }
        }};
    }

    #[test]
    fn state_logp_values_single_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0];
        let vals = vec![vec![Datum::Continuous(1.2)]];
        let logp = state_logp(
            &states[0],
            &col_ixs,
            &vals,
            &Given::Nothing,
            None,
            None,
        );

        assert_relative_eq!(logp[0], -2.939_618_577_673_343_7, epsilon = TOL);
    }

    #[test]
    fn state_logp_vs_exp_values_single_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0];
        let vals = vec![vec![Datum::Continuous(1.2)]];
        state_logp_vs_exp!(false, &states[0], &col_ixs, &vals, &Given::Nothing);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(-0.3)]];
        let logp = state_logp(
            &states[0],
            &col_ixs,
            &vals,
            &Given::Nothing,
            None,
            None,
        );

        assert_relative_eq!(logp[0], -4.277_889_544_469_348, epsilon = TOL);
    }

    #[test]
    fn state_logp_vs_exp_values_multi_col_single_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(-0.3)]];
        state_logp_vs_exp!(false, &states[0], &col_ixs, &vals, &Given::Nothing)
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
            None,
        );

        assert_relative_eq!(logp[0], -4.277_889_544_469_348, epsilon = TOL);
    }

    #[test]
    fn state_logp_vs_exp_values_multi_col_single_view_precomp() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(-0.3)]];

        state_logp_vs_exp!(true, &states[0], &col_ixs, &vals, &Given::Nothing);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(0.2)]];
        let logp = state_logp(
            &states[0],
            &col_ixs,
            &vals,
            &Given::Nothing,
            None,
            None,
        );

        assert_relative_eq!(logp[0], -4.718_619_899_900_069, epsilon = TOL);
    }

    #[test]
    fn state_logp_vs_exp_values_multi_col_multi_view() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(0.2)]];
        state_logp_vs_exp!(false, &states[0], &col_ixs, &vals, &Given::Nothing);
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
            None,
        );

        assert_relative_eq!(logp[0], -4.718_619_899_900_069, epsilon = TOL);
    }

    #[test]
    fn state_logp_vs_exp_values_multi_col_multi_view_precomp() {
        let states = get_states_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![Datum::Continuous(1.2), Datum::Continuous(0.2)]];
        state_logp_vs_exp!(true, &states[0], &col_ixs, &vals, &Given::Nothing);
    }

    #[test]
    fn single_state_continuous_impute_1() {
        let mut all_states = get_states_from_yaml();
        let states = [&all_states.remove(0)];
        let x: f64 = continuous_impute(&states, 1, 0);
        assert_relative_eq!(x, 1.683_113_796_266_261_7, epsilon = 10E-6);
    }

    #[test]
    fn single_state_continuous_impute_2() {
        let mut all_states = get_states_from_yaml();
        let states = [&all_states.remove(0)];
        let x: f64 = continuous_impute(&states, 3, 0);
        assert_relative_eq!(x, -0.824_416_188_399_796_6, epsilon = 10E-6);
    }

    #[test]
    fn multi_state_continuous_impute_1() {
        let mut all_states = get_states_from_yaml();
        let states = [&all_states.remove(0), &all_states.remove(0)];
        let x: f64 = continuous_impute(&states, 1, 2);
        assert_relative_eq!(x, 0.554_604_492_187_499_9, epsilon = 10E-6);
    }

    #[test]
    fn multi_state_continuous_impute_2() {
        let states = get_states_from_yaml();
        let states: Vec<&State> = states.iter().collect();
        let x: f64 = continuous_impute(&states, 1, 2);
        assert_relative_eq!(x, -0.250_584_379_015_657_5, epsilon = 10E-6);
    }

    #[test]
    fn single_state_categorical_impute_1() {
        let state: State = get_single_categorical_state_from_yaml();
        let x: u8 = categorical_impute(&[&state], 0, 0);
        assert_eq!(x, 2);
    }

    #[test]
    fn single_state_categorical_impute_2() {
        let state: State = get_single_categorical_state_from_yaml();
        let x: u8 = categorical_impute(&[&state], 2, 0);
        assert_eq!(x, 0);
    }

    #[test]
    fn single_state_categorical_predict_1() {
        let state: State = get_single_categorical_state_from_yaml();
        let x: u8 = categorical_predict(&[&state], 0, &Given::Nothing);
        assert_eq!(x, 2);
    }

    #[test]
    fn single_state_categorical_entropy() {
        let state: State = get_single_categorical_state_from_yaml();
        let h = entropy_single(0, &vec![state]);
        assert_relative_eq!(h, 1.368_541_708_152_32, epsilon = 10E-6);
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
        assert_relative_eq!(h_x, 1.368_715_500_467_195_1, epsilon = 1E-12);
    }

    #[test]
    fn multi_state_categorical_single_entropy_vs_old() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        for col_ix in 0..oracle.n_cols() {
            let h_x_new = entropy_single(col_ix, &oracle.states);
            let h_x_old =
                old_categorical_entropy_single(col_ix, &oracle.states);
            assert_relative_eq!(h_x_new, h_x_old, epsilon = 1E-12);
        }
    }

    #[test]
    fn single_state_labeler_impute_2() {
        let state: State = get_single_labeler_state_from_yaml();
        let x: Label = labeler_impute(&[&state], 9, 0);
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
        let x: Label = labeler_impute(&[&state], 1, 0);
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
        let states = [&get_single_count_state_from_yaml()];
        let x: u32 = count_impute(&states, 1, 0);
        assert_eq!(x, 1);
    }

    #[test]
    fn single_state_count_impute_2() {
        let states = [&get_single_count_state_from_yaml()];
        let x: u32 = count_impute(&states, 1, 0);
        assert_eq!(x, 1);
    }

    #[test]
    fn single_state_count_predict() {
        let states = [&get_single_count_state_from_yaml()];
        let x: u32 = count_predict(&states, 0, &Given::Nothing);
        assert_eq!(x, 1);
    }

    #[test]
    fn single_state_dual_categorical_entropy_0() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.drain(..).next().unwrap();
        let hxy = categorical_entropy_dual(2, 3, &vec![state]);
        assert_relative_eq!(hxy, 2.050_396_319_359_273_4, epsilon = 1E-14);
    }

    #[test]
    fn single_state_dual_categorical_entropy_1() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.pop().unwrap();
        let hxy = categorical_entropy_dual(2, 3, &vec![state]);
        assert_relative_eq!(hxy, 2.035_433_971_709_626, epsilon = 1E-14);
    }

    #[test]
    fn single_state_dual_categorical_entropy_vs_joint_equiv() {
        let states = {
            let mut states = get_entropy_states_from_yaml();
            let state = states.pop().unwrap();
            vec![state]
        };
        let hxy_dual = categorical_entropy_dual(2, 3, &states);
        let hxy_joint = categorical_joint_entropy(&[2, 3], &states);

        assert_relative_eq!(hxy_dual, hxy_joint, epsilon = 1E-14);
    }

    #[test]
    fn multi_state_dual_categorical_entropy_1() {
        let states = get_entropy_states_from_yaml();
        let hxy = categorical_entropy_dual(2, 3, &states);
        assert_relative_eq!(hxy, 2.050_402_245_628_641_5, epsilon = 1E-14);
    }

    #[test]
    fn multi_state_dual_categorical_entropy_vs_joint_equiv() {
        let states = get_entropy_states_from_yaml();
        let hxy_dual = categorical_entropy_dual(2, 3, &states);
        let hxy_joint = categorical_joint_entropy(&[2, 3], &states);
        assert_relative_eq!(hxy_dual, hxy_joint, epsilon = 1E-14);
    }

    #[test]
    fn single_state_categorical_gaussian_entropy_0() {
        let mut states = get_entropy_states_from_yaml();
        // first state
        let state = states.drain(..).next().unwrap();
        let hxy = categorical_gaussian_entropy_dual(2, 0, &vec![state]);
        assert_relative_eq!(hxy, 2.726_163_712_601_034, epsilon = 1E-7);
    }

    #[test]
    fn single_state_categorical_gaussian_entropy_1() {
        let mut states = get_entropy_states_from_yaml();
        // second (last) state
        let state = states.pop().unwrap();
        let hxy = categorical_gaussian_entropy_dual(2, 0, &vec![state]);
        assert_relative_eq!(hxy, 2.735_457_532_371_074_6, epsilon = 1E-7);
    }

    #[test]
    fn multi_state_categorical_gaussian_entropy_0() {
        let states = get_entropy_states_from_yaml();
        let hxy = categorical_gaussian_entropy_dual(2, 0, &states);
        assert_relative_eq!(hxy, 2.744_356_173_055_859, epsilon = 1E-7);
    }

    #[test]
    fn sobol_samples() {
        let mut states = get_entropy_states_from_yaml();
        let state = states.pop().unwrap();
        let (samples, _) = gen_sobol_samples(&[0, 2, 3], &state, 102);

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
            let (samples, q_recip) = gen_sobol_samples(&[col_ix], &state, n);

            let logps = state_logp(
                &state,
                &[col_ix],
                &samples,
                &Given::Nothing,
                None,
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
        assert_relative_eq!(h_exact, h_sobol, epsilon = 1E-7);
    }
}

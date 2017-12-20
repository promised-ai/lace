extern crate itertools;
extern crate rand;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::io::Read;
use std::iter::FromIterator;

use self::rand::Rng;
use self::itertools::multizip;

use cc::DType;
use cc::State;
use cc::ColModel;
use dist::Categorical;
use dist::traits::{RandomVariate, KlDivergence};
use misc::logsumexp;


/// Teller answers questions
pub struct Teller {
    /// Vector of data-less states
    pub states: Vec<State>,
}


impl Teller {
    pub fn new(states: Vec<State>) -> Self {
        Teller{states: states}
    }

    /// Build a teller from a list of yaml files
    pub fn from_yaml(filenames: Vec<&str>) -> Self {
        // TODO: Input validation
        // TODO: Should return Result<Self>
        let states = filenames.iter().map(|filename| {
            let path = Path::new(&filename);
            let mut file = File::open(&path).unwrap();
            let mut yaml = String::new();
            file.read_to_string(&mut yaml);
            serde_yaml::from_str(&yaml).unwrap()
        }).collect();

        Teller{states: states}
    }

    /// Returns the number of stats in the `Teller`
    pub fn nstates(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of rows in the `Teller`
    pub fn nrows(&self) -> usize {
        self.states[0].nrows()
    }

    /// Returns the number of columns/features in the `Teller`
    pub fn ncols(&self) -> usize {
        self.states[0].ncols()
    }

    /// Estimated dependence probability between `col_a` and `col_b`
    pub fn depprob(&self, col_a: usize, col_b: usize) -> f64 {
        self.states.iter().fold(0.0, |acc, state| {
            if state.asgn.asgn[col_a] == state.asgn.asgn[col_b] {
                acc + 1.0
            } else {
                acc
            }
        }) / (self.nstates() as f64)
    }

    /// Estimated row similarity between `row_a` and `row_b`
    pub fn rowsim(&self, row_a: usize, row_b: usize,
                  wrt: Option<&Vec<usize>>) -> f64
    {
        self.states.iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt {
                Some(col_ixs) => {
                    let asgn = &state.asgn.asgn;
                    let viewset: HashSet<usize> = HashSet::from_iter(
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]));
                    viewset.iter().map(|x| *x).collect()
                },
                None => (0..state.views.len()).collect(),
            };
            acc + view_ixs.iter().fold(0.0, |sim, &view_ix| {
                let asgn = &state.views[view_ix].asgn.asgn;
                if asgn[row_a] == asgn[row_b] {
                    sim + 1.0
                } else {
                    sim
                }
            }) / (view_ixs.len() as f64)
        }) / self.nstates() as f64
    }

    /// Exstimate the mutual information between col_a and col_b using Monte
    /// Carlo integration
    pub fn mutual_information(&self, col_a: usize, col_b: usize, n: usize,
                              mut rng: &mut Rng) -> f64
    {
        // TODO: Would be a lot faster if logp took a vector of values. Here,
        // we're having to repeadedly recompute the exact same weights for each
        // call of logp().
        let col_ixs = vec![col_a, col_b];

        let vals_ab = self.simulate(&col_ixs, &None, n, &mut rng);
        let vals_a = vals_ab.iter().map(|vals| vec![vals[0].clone()]).collect();
        let vals_b = vals_ab.iter().map(|vals| vec![vals[1].clone()]).collect();

        let logps_ab = self.logp(&col_ixs, &vals_ab,  &None);
        let logps_a = self.logp(&vec![col_a], &vals_a, &None);
        let logps_b = self.logp(&vec![col_b], &vals_b, &None);

        multizip((&logps_ab, &logps_a, &logps_b))
            .fold(0.0, |acc, (ab, a, b)| {
                acc + ab - a - b
            }) - (n as f64).ln()
    }

    /// Estimate entropy using Monte Carlo integration
    pub fn entropy(&self, col_ixs: &Vec<usize>, n: usize, mut rng: &mut Rng)
        -> f64
    {
        let log_n = (n as f64).ln();

        let vals = self.simulate(&col_ixs, &None, n, &mut rng);
        self.logp(&col_ixs, &vals, &None)
            .iter()
            .fold(0.0, |acc, logp| logp + acc) - log_n
    }

    /// Conditional entropy H(A|B)
    pub fn conditional_entropy(&self, col_a: usize, col_b: usize) -> f64 {
        unimplemented!();
    }

    /// Negative log PDF/PMF of x in row_ix, col_ix
    pub fn surprisal(&self, x: &DType, row_ix: usize, col_ix: usize) -> f64 {
        let logps: Vec<f64> = self.states.iter().map(|state| {
            let view_ix = state.asgn.asgn[col_ix];
            let k = state.views[view_ix].asgn.asgn[row_ix];
            state.views[view_ix].ftrs[&col_ix].cpnt_logp(x, k)
        }).collect();
        - logsumexp(&logps) + (self.nstates() as f64).ln()
    }

    // TODO: Should take vector of vectors and compute multiple probabilities
    // to save recomputing the weights.
    pub fn logp(&self, col_ixs: &Vec<usize>,
                vals: &Vec<Vec<DType>>, given_opt: &Option<Vec<(usize, DType)>>)
                -> Vec<f64>
    {
        let n = vals.len();
        let mut logp_sum: Vec<f64> = self.states
            .iter()
            .map(|state| state_logp(state, &col_ixs, &vals, &given_opt))
            .fold(vec![0.0; n], |mut acc, logps| {
                acc.iter_mut().zip(logps).for_each(|(ac, lp)| *ac += lp);
                acc
            });

        let log_nstates = (self.nstates() as f64).ln();
        logp_sum.iter_mut().for_each(|lp| *lp -= log_nstates);
        logp_sum
    }

    /// Simulate values from joint or conditional distribution
    pub fn simulate(
        &self, col_ixs: &Vec<usize>,
        given_opt: &Option<Vec<(usize, DType)>>,
        n: usize,
        mut rng: &mut Rng
        ) -> Vec<Vec<DType>>
    {
        let weights = given_weights(&self.states, &col_ixs, &given_opt);
        let state_ixer = Categorical::flat(self.nstates());

        (0..n).map(|_| {
            // choose a random state
            let state_ix: usize = state_ixer.draw(&mut rng);
            let state = &self.states[state_ix];

            // for each view
            //   choose a random component from the weights
            let mut cpnt_ixs: BTreeMap<usize, usize> = BTreeMap::new();
            for (view_ix, view_weights) in &weights[state_ix] {
                let component_ixer = Categorical::new(view_weights.clone());
                let k = component_ixer.draw(&mut rng);
                cpnt_ixs.insert(*view_ix, k);
            }

            // for eacch column
            //   draw from appropriate component from that view
            let mut xs: Vec<DType> = Vec::with_capacity(col_ixs.len());
            col_ixs.iter().for_each(|col_ix| {
                let view_ix = state.asgn.asgn[*col_ix];
                let k = cpnt_ixs[&view_ix];
                let x = state.views[view_ix].ftrs[col_ix].draw(k, &mut rng);
                xs.push(x);
            });
            xs
        }).collect()
    }

    // TODO: may offload to a python function?
    pub fn predict(&self, row_ix: usize, col_ix: usize) -> (DType, f64) {
        unimplemented!();
    }

    // TODO: Use JS Divergence?
    // TODO: Use 1 - KL and reframe as certainty?
    /// Computes the predictive uncertainty for the datum at (row_ix, col_ix)
    /// as mean the pairwise KL divergence between the components to which the
    /// datum is assigned.
    pub fn predictive_uncertainty(&self, row_ix: usize, col_ix: usize) -> f64 {
        let locators: Vec<(usize, usize)> = self.states
            .iter()
            .map(|state| {
            let view_ix = state.asgn.asgn[col_ix];
            let cpnt_ix = state.views[view_ix].asgn.asgn[row_ix];
            (view_ix, cpnt_ix)
        }).collect();

        // FIXME: this code makes me want to die
        let mut kl_sum = 0.0;
        for (i, &(vi, ki)) in locators.iter().enumerate() {
            let cm_i = &self.states[i].views[vi].ftrs[&col_ix];
            match cm_i {
                &ColModel::Continuous(ref fi)  => {
                    let cpnt_i = &fi.components[ki];
                    for (j, &(vj, kj)) in locators.iter().enumerate() {
                        if i != j {
                            let cm_j = &self.states[j].views[vj].ftrs[&col_ix];
                            match cm_j {
                                &ColModel::Continuous(ref fj) => {
                                    let cpnt_j = &fj.components[kj];
                                    kl_sum += cpnt_i.kl_divergence(cpnt_j);
                                },
                                _ => panic!("2nd ColModel was not continuous"),
                            }
                        }
                    }
                },
                &ColModel::Categorical(ref fi)  => {
                    let cpnt_i = &fi.components[ki];
                    for (j, &(vj, kj)) in locators.iter().enumerate() {
                        if i != j {
                            let cm_j = &self.states[j].views[vj].ftrs[&col_ix];
                            match cm_j {
                                &ColModel::Categorical(ref fj) => {
                                    let cpnt_j = &fj.components[kj];
                                    kl_sum += cpnt_i.kl_divergence(cpnt_j);
                                },
                                _ => panic!("2nd ColModel was not categorical"),
                            }
                        }
                    }
                },
            }
        }

        let nstates = self.nstates() as f64;
        kl_sum / (nstates * nstates - nstates)
    }
}


// Helper functions
// ================
// Weight Calculation
// ------------------
fn given_weights(
    states: &Vec<State>, col_ixs: &Vec<usize>,
    given_opt: &Option<Vec<(usize, DType)>>)
    -> Vec<BTreeMap<usize, Vec<f64>>>
{
    let mut state_weights: Vec<_> = Vec::with_capacity(states.len());

    for state in states {
        let view_weights = single_state_weights(
            &state, &col_ixs, &given_opt, false);
        state_weights.push(view_weights);
    }
    state_weights
}


fn single_state_weights(state: &State, col_ixs: &Vec<usize>,
                        given_opt: &Option<Vec<(usize, DType)>>,
                        weightless: bool) -> BTreeMap<usize, Vec<f64>>
{
    let mut view_ixs: HashSet<usize> = HashSet::new();
    col_ixs.iter().for_each(|col_ix| {
        let view_ix = state.asgn.asgn[*col_ix];
        view_ixs.insert(view_ix);
    });

    let mut view_weights: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    view_ixs.iter()
        .for_each(|&view_ix| {
            let weights = single_view_weights(&state, view_ix, &given_opt,
                                              weightless);
            view_weights.insert(view_ix, weights);
        });
    view_weights
}


fn single_view_weights(state: &State, target_view_ix: usize,
                       given_opt: &Option<Vec<(usize, DType)>>,
                       weightless: bool) -> Vec<f64> {
    let view = &state.views[target_view_ix];

    let mut weights = if weightless {
        vec![0.0; view.asgn.ncats]
    } else {
        view.asgn.log_weights()
    };

    match given_opt {
        &Some(ref given) => {
            for &(id, ref datum) in given {
                let in_target_view = state.asgn.asgn[id] == target_view_ix;
                if in_target_view {
                    weights = view.ftrs[&id].accum_weights(&datum, weights);
                }
            }},
        &None => (),
    }
    weights
}


// Probability calculation
// -----------------------
fn state_logp(state: &State, col_ixs: &Vec<usize>,
              vals: &Vec<Vec<DType>>, given_opt: &Option<Vec<(usize, DType)>>)
              -> Vec<f64>
{
    let mut view_weights = single_state_weights(state, &col_ixs, &given_opt, false);

    // normalize view weights
    for weights in view_weights.values_mut() {
        let logz = logsumexp(weights);
        weights.iter_mut().for_each(|w| *w -= logz);
    }

    vals.iter()
        .map(|val| single_val_logp(&state, &col_ixs, &val, &view_weights))
        .collect()
}


fn single_val_logp(state: &State, col_ixs: &Vec<usize>, val: &Vec<DType>,
                   view_weights: &BTreeMap<usize, Vec<f64>>) -> f64 {
    // turn col_ixs and values into a given
    let mut obs = Vec::with_capacity(col_ixs.len());
    for (&col_ix, datum) in col_ixs.iter().zip(val) {
        obs.push((col_ix, datum.clone()));
    }

    // compute the un-normalied, 'weightless', weights using the given
    let mut logp_obs = single_state_weights(state, &col_ixs, &Some(obs), true);

    // add everything up
    let mut logp_out = 0.0;
    for (view_ix, mut logps) in logp_obs {
        let weights = &view_weights[&view_ix];
        logps.iter_mut().zip(weights.iter()).for_each(|(lp, w)| *lp += w);
        logp_out += logsumexp(&logps); 
    }
    logp_out
}


#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;

    const TOL: f64 = 1E-8;

    fn get_teller_from_yaml() -> Teller {
        let filenames = vec![
            "resources/test/small-state-1.yaml",
            "resources/test/small-state-2.yaml",
            "resources/test/small-state-3.yaml"];

        Teller::from_yaml(filenames)
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let teller = get_teller_from_yaml();

        let weights_0 = single_view_weights(&teller.states[0], 0, &None, false);

        assert_relative_eq!(weights_0[0], -0.6931471805599453, epsilon=TOL);
        assert_relative_eq!(weights_0[1], -0.6931471805599453, epsilon=TOL);

        let weights_1 = single_view_weights(&teller.states[0], 1, &None, false);

        assert_relative_eq!(weights_1[0], -1.3862943611198906, epsilon=TOL);
        assert_relative_eq!(weights_1[1], -0.2876820724517809, epsilon=TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_one_given() {
        let teller = get_teller_from_yaml();

        // column 1 should not affect view 0 weights because it is assigned to
        // view 1
        let given = Some(vec![(0, DType::Continuous(0.0)),
                              (1, DType::Continuous(-1.0))]);

        let weights_0 = single_view_weights(&teller.states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -3.1589583681201292, epsilon=TOL);
        assert_relative_eq!(weights_0[1], -1.9265784475169849, epsilon=TOL);

        let weights_1 = single_view_weights(&teller.states[0], 1, &given, false);

        assert_relative_eq!(weights_1[0], -4.0958633027669231, epsilon=TOL);
        assert_relative_eq!(weights_1[1], -0.4177811369331429, epsilon=TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_added_given() {
        let teller = get_teller_from_yaml();

        let given = Some(vec![(0, DType::Continuous(0.0)),
                              (2, DType::Continuous(-1.0))]);

        let weights_0 = single_view_weights(&teller.states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -5.6691757676902537, epsilon=TOL);
        assert_relative_eq!(weights_0[1], -9.3045547861934459, epsilon=TOL);
    }

    #[test]
    fn single_state_weights_value_check() {
        let teller = get_teller_from_yaml();

        let state = &teller.states[0];
        let col_ixs = vec![0, 1];
        let given = Some(vec![(0, DType::Continuous(0.0)),
                              (1, DType::Continuous(-1.0)),
                              (2, DType::Continuous(-1.0))]);

        let weights = single_state_weights(state, &col_ixs, &given, false);

        assert_eq!(weights.len(), 2);
        assert_eq!(weights[&0].len(), 2);
        assert_eq!(weights[&1].len(), 2);

        assert_relative_eq!(weights[&0][0], -5.6691757676902537, epsilon=TOL);
        assert_relative_eq!(weights[&0][1], -9.3045547861934459, epsilon=TOL);

        assert_relative_eq!(weights[&1][0], -4.0958633027669231, epsilon=TOL);
        assert_relative_eq!(weights[&1][1], -0.4177811369331429, epsilon=TOL);
    }

    #[test]
    fn give_weights_size_check_single_target_column() {
        let teller = get_teller_from_yaml();

        let col_ixs = vec![0];
        let state_weights = given_weights(&teller.states, &col_ixs, &None);

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
        let teller = get_teller_from_yaml();

        let col_ixs = vec![0];
        let vals = vec![vec![DType::Continuous(1.2)]];
        let logp = state_logp(&teller.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -2.9396185776733437, epsilon=TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let teller = get_teller_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![DType::Continuous(1.2), DType::Continuous(-0.3)]];
        let logp = state_logp(&teller.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -4.2778895444693479, epsilon=TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let teller = get_teller_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![DType::Continuous(1.2), DType::Continuous(0.2)]];
        let logp = state_logp(&teller.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -4.7186198999000686, epsilon=TOL);
    }

    #[test]
    fn mutual_information_smoke() {
        let teller = get_teller_from_yaml();
        let mut rng = rand::thread_rng();

        let mi_01 = teller.mutual_information(0, 1, 1000, &mut rng);
        let mi_02 = teller.mutual_information(0, 2, 1000, &mut rng);
        let mi_12 = teller.mutual_information(1, 2, 1000, &mut rng);

        assert!(mi_01 > 0.0);
        assert!(mi_02 > 0.0);
        assert!(mi_12 > 0.0);
    }

    #[test]
    fn surpisal_value() {
        let teller = get_teller_from_yaml();
        let s = teller.surprisal(&DType::Continuous(1.2), 3, 1);
        assert_relative_eq!(s, 1.7739195803316758, epsilon=10E-7);
    }

    #[test]
    fn uncertainty_smoke() {
        let teller = get_teller_from_yaml();
        let u = teller.predictive_uncertainty(0, 1);
        // assert_relative_eq!(s, 1.7739195803316758, epsilon=10E-7);
    }
}

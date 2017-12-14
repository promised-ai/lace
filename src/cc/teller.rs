extern crate rand;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::io::Read;
use std::iter::FromIterator;

use self::rand::Rng;

use cc::DType;
use cc::State;
use dist::Categorical;
use dist::traits::RandomVariate;
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

    pub fn nstates(&self) -> usize {
        self.states.len()
    }

    pub fn nrows(&self) -> usize {
        self.states[0].nrows()
    }

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
        let h_sum: f64 = self
            .simulate(&col_ixs, &None, n, &mut rng)
            .iter()
            .map(|vals| {
                let a = vals[0].clone();
                let b = vals[1].clone();
                let vals_a = vec![a.clone()];
                let vals_b = vec![b.clone()];
                let vals_ab = vec![a, b];

                let logp_ab = self.logp(&col_ixs, &vals_ab,  &None);
                let logp_a = self.logp(&vec![col_a], &vals_a, &None);
                let logp_b = self.logp(&vec![col_b], &vals_b, &None);

                logp_ab - logp_a - logp_b })
            .sum();
        h_sum - (n as f64).ln()
    }

    /// Estimate entropy using Monte Carlo integration
    pub fn entropy(&self, col_ixs: &Vec<usize>, n: usize, mut rng: &mut Rng)
        -> f64
    {
        let log_n = (n as f64).ln();

        self.simulate(&col_ixs, &None, n, &mut rng)
            .iter()
            .map(|vals| self.logp(&col_ixs, &vals, &None) - log_n)
            .sum()
    }

    /// Conditional entropy H(A|B)
    pub fn conditional_entropy(&self, col_a: usize, col_b: usize) -> f64 {
        unimplemented!();
    }

    /// Negative log PDF/PMF of x in row, col
    pub fn surprisal(&self, x: &DType, row: usize, col: usize) -> f64 {
        unimplemented!();
    }

    // TODO: Should take vector of vectors and compute multiple probabilities
    // to save recomputing the weights.
    pub fn logp(&self, col_ixs: &Vec<usize>,
                vals: &Vec<DType>, given_opt: &Option<Vec<(usize, DType)>>)
                -> f64
    {
       let logp_sum: f64 = self.states
           .iter()
           .map(|state| state_logp(state, &col_ixs, &vals, &given_opt))
           .sum();

        logp_sum - (self.nstates() as f64).ln()
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

    pub fn predict(&self, row_ix: usize, col_ix: usize) -> (DType, f64) {
        unimplemented!();
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
              vals: &Vec<DType>, given_opt: &Option<Vec<(usize, DType)>>)
              -> f64
{
    let mut view_weights = single_state_weights(state, &col_ixs, &given_opt, false);

    // normalize view weights
    for weights in view_weights.values_mut() {
        let logz = logsumexp(weights);
        weights.iter_mut().for_each(|w| *w -= logz);
    }

    // turn col_ixs and values into a given
    let mut obs = Vec::with_capacity(col_ixs.len());
    for (&col_ix, datum) in col_ixs.iter().zip(vals) {
        obs.push((col_ix, datum.clone()));
    }

    // compute the un-normalied, 'weightless', weights using the given
    let logp_obs = single_state_weights(state, &col_ixs, &Some(obs), true);

    // add everything up
    let mut logp_out = 0.0;
    for (view_ix, mut weights) in view_weights {
        let logps = &logp_obs[&view_ix];
        weights.iter_mut().zip(logps.iter()).for_each(|(w, lp)| *w += lp);
        logp_out += logsumexp(&weights); 
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
        let vals = vec![DType::Continuous(1.2)];
        let logp: f64 = state_logp(&teller.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp, -2.9396185776733437, epsilon=TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let teller = get_teller_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![DType::Continuous(1.2), DType::Continuous(-0.3)];
        let logp: f64 = state_logp(&teller.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp, -4.2778895444693479, epsilon=TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let teller = get_teller_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![DType::Continuous(1.2), DType::Continuous(0.2)];
        let logp: f64 = state_logp(&teller.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp, -4.7186198999000686, epsilon=TOL);
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
}

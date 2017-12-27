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

use cc::DType;
use cc::State;
use cc::ColModel;
use dist::Categorical;
use dist::MixtureModel;
use dist::traits::{RandomVariate, KlDivergence};
use misc::logsumexp;


/// Oracle answers questions
pub struct Oracle {
    /// Vector of data-less states
    pub states: Vec<State>,
}


/// Mutual Information Type
pub enum MiType {
    /// The Standard, un-normalized variant
    UnNormed,
    /// Linfoot information Quantity. Derived by computing the mutual
    /// information between the two components of a bivariate Normal with
    /// covariance rho, and solving for rho.
    Linfoot,
    /// Variation of Information. A version of mutual information that
    /// satisfies the triangle inequality.
    Voi,
    /// Variation of Information normalized to [0, 1].
    NormedVoi,
    /// Information Quality Ratio:  the amount of information of a variable
    /// based on another variable against total uncertainty.
    Iqr,
    /// Mutual Information normed the with square root of the product of the
    /// components entropies. Akin to the Pearson correlation coefficient.
    Pearson,
}


impl Oracle {
    pub fn new(states: Vec<State>) -> Self {
        Oracle{states: states}
    }

    /// Load a Oracle from YAML, MessagePack, or JSON.
    pub fn load(path: &Path) -> Self {
        unimplemented!();
    }

    /// Build a oracle from a list of yaml files
    pub fn from_yaml(filenames: Vec<&str>) -> Self {
        // TODO: Input validation
        // TODO: Should return Result<Self>
        let states = filenames.iter().map(|filename| {
            let path = Path::new(&filename);
            let mut file = File::open(&path).unwrap();
            let mut yaml = String::new();
            let res = file.read_to_string(&mut yaml);
            match res {
                Ok(_)    => serde_yaml::from_str(&yaml).unwrap(),
                Err(err) => panic!("Error: {:?}", err),
            }
        }).collect();

        Oracle{states: states}
    }

    /// Returns the number of stats in the `Oracle`
    pub fn nstates(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of rows in the `Oracle`
    pub fn nrows(&self) -> usize {
        self.states[0].nrows()
    }

    /// Returns the number of columns/features in the `Oracle`
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

    /// Estimate the mutual information between col_a and col_b using Monte
    /// Carlo integration
    pub fn mutual_information(&self, col_a: usize, col_b: usize, n: usize,
                              mi_type: MiType, mut rng: &mut Rng) -> f64
    {
        let col_ixs = vec![col_a, col_b];

        let vals_ab = self.simulate(&col_ixs, &None, n, &mut rng);
        let vals_a = vals_ab.iter().map(|vals| vec![vals[0].clone()]).collect();
        let vals_b = vals_ab.iter().map(|vals| vec![vals[1].clone()]).collect();

        let h_ab = self.entropy_from_samples(&vals_ab, &col_ixs);
        let h_a = self.entropy_from_samples(&vals_a, &vec![col_a]);
        let h_b = self.entropy_from_samples(&vals_b, &vec![col_b]);

        // https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants
        match mi_type {
            MiType::UnNormed  => h_a + h_b - h_ab,
            MiType::Voi       => h_a + h_b - 2.0*h_ab,
            MiType::NormedVoi => (h_a + h_b - 2.0*h_ab) / h_ab,
            MiType::Iqr       => (h_a + h_b - h_ab) / h_ab,
            MiType::Pearson   => (h_a + h_b - h_ab) / (h_a * h_b).sqrt(),
            MiType::Linfoot   => {
                let mi = h_a + h_b - h_ab;
                (1.0 - (-2.0 * mi).exp()).sqrt()
            },
        }
    }

    /// Estimate entropy using Monte Carlo integration
    pub fn entropy(&self, col_ixs: &Vec<usize>, n: usize, mut rng: &mut Rng)
        -> f64
    {
        let vals = self.simulate(&col_ixs, &None, n, &mut rng);
        self.entropy_from_samples(&vals, &col_ixs)
    }

    fn entropy_from_samples(&self, vals: &Vec<Vec<DType>>, col_ixs: &Vec<usize>)
        -> f64
    {
        let log_n = (vals.len() as f64).ln();
        self.logp(&col_ixs, &vals, &None)
            .iter()
            .fold(0.0, |acc, logp| acc - logp) - log_n
    }

    /// Conditional entropy H(A|B)
    pub fn conditional_entropy(&self, _col_a: usize, _col_b: usize) -> f64 {
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
    pub fn predict(&self, _row_ix: usize, _col_ix: usize) -> (DType, f64) {
        unimplemented!();
    }

    // TODO: Use JS Divergence?
    // TODO: Use 1 - KL and reframe as certainty?
    /// Computes the predictive uncertainty for the datum at (row_ix, col_ix)
    /// as mean the pairwise KL divergence between the components to which the
    /// datum is assigned.
    pub fn predictive_uncertainty(&self, row_ix: usize, col_ix: usize,
                                  n_samples: usize, mut rng: &mut Rng) -> f64 {
        if n_samples > 0 {
            js_uncertainty(&self.states, row_ix, col_ix, n_samples, &mut rng)
        } else {
            kl_uncertainty(&self.states, row_ix, col_ix)
        }
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
    let logp_obs = single_state_weights(state, &col_ixs, &Some(obs), true);

    // add everything up
    let mut logp_out = 0.0;
    for (view_ix, mut logps) in logp_obs {
        let weights = &view_weights[&view_ix];
        logps.iter_mut().zip(weights.iter()).for_each(|(lp, w)| *lp += w);
        logp_out += logsumexp(&logps); 
    }
    logp_out
}


// Predictive uncertainty helpers
// ------------------------------
// FIXME: this code also makes me want to kill myself
fn js_uncertainty(states: &Vec<State>, row_ix: usize, col_ix: usize,
                  n_samples: usize, mut rng: &mut Rng) -> f64 {
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
                    },
                    _ => panic!("Mismatched feature type"),
                }
            }
            let m = MixtureModel::flat(cpnts);
            m.js_divergence(n_samples, &mut rng)
        },
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
                    },
                    _ => panic!("Mismatched feature type"),
                }
            }
            let m = MixtureModel::flat(cpnts);
            m.js_divergence(n_samples, &mut rng)
        },
        
    }
}


pub fn kl_uncertainty(states: &Vec<State>, row_ix: usize, col_ix: usize) -> f64 {
    let locators: Vec<(usize, usize)> = states
        .iter()
        .map(|state| {
        let view_ix = state.asgn.asgn[col_ix];
        let cpnt_ix = state.views[view_ix].asgn.asgn[row_ix];
        (view_ix, cpnt_ix)
    }).collect();

    // FIXME: this code makes me want to die
    let mut kl_sum = 0.0;
    for (i, &(vi, ki)) in locators.iter().enumerate() {
        let cm_i = &states[i].views[vi].ftrs[&col_ix];
        match cm_i {
            &ColModel::Continuous(ref fi)  => {
                let cpnt_i = &fi.components[ki];
                for (j, &(vj, kj)) in locators.iter().enumerate() {
                    if i != j {
                        let cm_j = &states[j].views[vj].ftrs[&col_ix];
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
                        let cm_j = &states[j].views[vj].ftrs[&col_ix];
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

    let nstates = states.len() as f64;
    kl_sum / (nstates * nstates - nstates)
}


#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;

    const TOL: f64 = 1E-8;

    fn get_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/small-state-1.yaml",
            "resources/test/small-state-2.yaml",
            "resources/test/small-state-3.yaml"];

        Oracle::from_yaml(filenames)
    }

    #[test]
    fn single_view_weights_state_0_no_given() {
        let oracle = get_oracle_from_yaml();

        let weights_0 = single_view_weights(&oracle.states[0], 0, &None, false);

        assert_relative_eq!(weights_0[0], -0.6931471805599453, epsilon=TOL);
        assert_relative_eq!(weights_0[1], -0.6931471805599453, epsilon=TOL);

        let weights_1 = single_view_weights(&oracle.states[0], 1, &None, false);

        assert_relative_eq!(weights_1[0], -1.3862943611198906, epsilon=TOL);
        assert_relative_eq!(weights_1[1], -0.2876820724517809, epsilon=TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_one_given() {
        let oracle = get_oracle_from_yaml();

        // column 1 should not affect view 0 weights because it is assigned to
        // view 1
        let given = Some(vec![(0, DType::Continuous(0.0)),
                              (1, DType::Continuous(-1.0))]);

        let weights_0 = single_view_weights(&oracle.states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -3.1589583681201292, epsilon=TOL);
        assert_relative_eq!(weights_0[1], -1.9265784475169849, epsilon=TOL);

        let weights_1 = single_view_weights(&oracle.states[0], 1, &given, false);

        assert_relative_eq!(weights_1[0], -4.0958633027669231, epsilon=TOL);
        assert_relative_eq!(weights_1[1], -0.4177811369331429, epsilon=TOL);
    }

    #[test]
    fn single_view_weights_state_0_with_added_given() {
        let oracle = get_oracle_from_yaml();

        let given = Some(vec![(0, DType::Continuous(0.0)),
                              (2, DType::Continuous(-1.0))]);

        let weights_0 = single_view_weights(&oracle.states[0], 0, &given, false);

        assert_relative_eq!(weights_0[0], -5.6691757676902537, epsilon=TOL);
        assert_relative_eq!(weights_0[1], -9.3045547861934459, epsilon=TOL);
    }

    #[test]
    fn single_state_weights_value_check() {
        let oracle = get_oracle_from_yaml();

        let state = &oracle.states[0];
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
        let oracle = get_oracle_from_yaml();

        let col_ixs = vec![0];
        let state_weights = given_weights(&oracle.states, &col_ixs, &None);

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
        let oracle = get_oracle_from_yaml();

        let col_ixs = vec![0];
        let vals = vec![vec![DType::Continuous(1.2)]];
        let logp = state_logp(&oracle.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -2.9396185776733437, epsilon=TOL);
    }

    #[test]
    fn state_logp_values_multi_col_single_view() {
        let oracle = get_oracle_from_yaml();

        let col_ixs = vec![0, 2];
        let vals = vec![vec![DType::Continuous(1.2), DType::Continuous(-0.3)]];
        let logp = state_logp(&oracle.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -4.2778895444693479, epsilon=TOL);
    }

    #[test]
    fn state_logp_values_multi_col_multi_view() {
        let oracle = get_oracle_from_yaml();

        let col_ixs = vec![0, 1];
        let vals = vec![vec![DType::Continuous(1.2), DType::Continuous(0.2)]];
        let logp = state_logp(&oracle.states[0], &col_ixs, &vals, &None);

        assert_relative_eq!(logp[0], -4.7186198999000686, epsilon=TOL);
    }

    #[test]
    fn mutual_information_smoke() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let mi_01 = oracle.mutual_information(0, 1, 1000, MiType::UnNormed,
                                              &mut rng);
        let mi_02 = oracle.mutual_information(0, 2, 1000, MiType::UnNormed,
                                              &mut rng);
        let mi_12 = oracle.mutual_information(1, 2, 1000, MiType::UnNormed,
                                              &mut rng);

        assert!(mi_01 > 0.0);
        assert!(mi_02 > 0.0);
        assert!(mi_12 > 0.0);
    }

    #[test]
    fn surpisal_value() {
        let oracle = get_oracle_from_yaml();
        let s = oracle.surprisal(&DType::Continuous(1.2), 3, 1);
        assert_relative_eq!(s, 1.7739195803316758, epsilon=10E-7);
    }

    #[test]
    fn kl_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();
        let u = oracle.predictive_uncertainty(0, 1, 0, &mut rng);

        assert!(u > 0.0);
    }
}
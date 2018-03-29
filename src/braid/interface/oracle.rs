extern crate csv;
extern crate itertools;
extern crate rand;
extern crate rmp_serde;
extern crate rusqlite;
extern crate serde_json;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::io::Result;
use std::iter::FromIterator;

use rayon::prelude::*;
use self::rand::Rng;

use cc::{Codebook, DType, FType, State};
use cc::DataStore;
use dist::Categorical;
use dist::traits::RandomVariate;
use misc::{logsumexp, transpose};
use interface::utils;
use cc::state::StateDiagnostics;
use interface::Given;
use cc::file_utils;

/// Oracle answers questions
#[derive(Clone, Serialize, Deserialize)]
pub struct Oracle {
    /// Vector of states
    pub states: Vec<State>,
    /// Metadata for the rows and columns
    pub codebook: Codebook,
    pub data: DataStore,
}

/// Mutual Information Type
#[derive(Serialize, Deserialize, Debug, Clone)]
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
    /// Jaccard distance betwwn X an Y. Jaccard(X, Y) is in [0, 1].
    Jaccard,
    /// Information Quality Ratio:  the amount of information of a variable
    /// based on another variable against total uncertainty.
    Iqr,
    /// Mutual Information normed the with square root of the product of the
    /// components entropies. Akin to the Pearson correlation coefficient.
    Pearson,
}

impl Oracle {
    /// Load an Oracle from a .braid file
    pub fn load(dir: &str) -> Result<Self> {
        let data = file_utils::load_data(dir)?;
        let mut states = file_utils::load_states(dir)?;
        let codebook = file_utils::load_codebook(dir)?;

        // Move states from map to vec
        let ids: Vec<usize> = states.keys().map(|k| *k).collect();
        let states_vec =
            ids.iter().map(|id| states.remove(id).unwrap()).collect();

        Ok(Oracle {
            states: states_vec,
            codebook: codebook,
            data: DataStore::new(data),
        })
    }

    pub fn state_diagnostics(&self) -> Vec<StateDiagnostics> {
        self.states
            .iter()
            .map(|state| state.diagnostics.clone())
            .collect()
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

    /// Returns a Vector of the feature types of each row
    pub fn ftypes(&self) -> Vec<FType> {
        (0..self.ncols()).map(|col_ix| self.ftype(col_ix)).collect()
    }

    pub fn ftype(&self, col_ix: usize) -> FType {
        let state = &self.states[0];
        let view_ix = state.asgn.asgn[col_ix];
        state.views[view_ix].ftrs[&col_ix].ftype()
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

    pub fn depprob_pw(&self, pairs: &Vec<(usize, usize)>) -> Vec<f64> {
        pairs
            .par_iter()
            .map(|(col_a, col_b)| self.depprob(*col_a, *col_b))
            .collect()
    }

    /// Estimated row similarity between `row_a` and `row_b`
    pub fn rowsim(
        &self,
        row_a: usize,
        row_b: usize,
        wrt: Option<&Vec<usize>>,
    ) -> f64 {
        self.states.iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt {
                Some(col_ixs) => {
                    let asgn = &state.asgn.asgn;
                    let viewset: HashSet<usize> = HashSet::from_iter(
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]),
                    );
                    viewset.iter().map(|x| *x).collect()
                }
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

    pub fn rowsim_pw(
        &self,
        pairs: &Vec<(usize, usize)>,
        wrt: Option<&Vec<usize>>,
    ) -> Vec<f64> {
        pairs
            .par_iter()
            .map(|(row_a, row_b)| self.rowsim(*row_a, *row_b, wrt.clone()))
            .collect()
    }

    /// Estimate the mutual information between col_a and col_b using Monte
    /// Carlo integration
    pub fn mi(
        &self,
        col_a: usize,
        col_b: usize,
        n: usize,
        mi_type: MiType,
        mut rng: &mut Rng,
    ) -> f64 {
        let col_ixs = vec![col_a, col_b];

        let vals_ab = self.simulate(&col_ixs, &None, n, &mut rng);
        // FIXME: Do these have to be simulated independently
        let vals_a = vals_ab.iter().map(|vals| vec![vals[0].clone()]).collect();
        let vals_b = vals_ab.iter().map(|vals| vec![vals[1].clone()]).collect();

        let h_ab = self.entropy_from_samples(&vals_ab, &col_ixs);
        let h_a = self.entropy_from_samples(&vals_a, &vec![col_a]);
        let h_b = self.entropy_from_samples(&vals_b, &vec![col_b]);

        // https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants
        match mi_type {
            MiType::UnNormed => h_a + h_b - h_ab,
            MiType::Voi => 2.0 * h_ab - h_a - h_b,
            MiType::Pearson => (h_a + h_b - h_ab) / (h_a * h_b).sqrt(),
            MiType::Iqr => (h_a + h_b - h_ab) / h_ab,
            MiType::Jaccard => 1.0 - (h_a + h_b - h_ab) / h_ab,
            MiType::Linfoot => {
                let mi = h_a + h_b - h_ab;
                (1.0 - (-2.0 * mi).exp()).sqrt()
            }
        }
    }

    pub fn mi_pw(
        &self,
        pairs: &Vec<(usize, usize)>,
        n: usize,
        mi_type: MiType,
        mut rng: &mut Rng,
    ) -> Vec<f64> {
        // TODO: Parallelize
        // TODO: Could save a lot of computation by memoizing the entopies
        pairs
            .iter()
            .map(|(col_a, col_b)| {
                self.mi(*col_a, *col_b, n, mi_type.clone(), &mut rng)
            })
            .collect()
    }

    /// Estimate entropy using Monte Carlo integration
    pub fn entropy(
        &self,
        col_ixs: &Vec<usize>,
        n: usize,
        mut rng: &mut Rng,
    ) -> f64 {
        let vals = self.simulate(&col_ixs, &None, n, &mut rng);
        self.entropy_from_samples(&vals, &col_ixs)
    }

    fn entropy_from_samples(
        &self,
        vals: &Vec<Vec<DType>>,
        col_ixs: &Vec<usize>,
    ) -> f64 {
        // let log_n = (vals.len() as f64).ln();
        self.logp(&col_ixs, &vals, &None)
            .iter()
            .fold(0.0, |acc, logp| acc - logp) / (vals.len() as f64)
    }

    /// Conditional entropy H(A|B)
    pub fn conditional_entropy(&self, _col_a: usize, _col_b: usize) -> f64 {
        unimplemented!();
    }

    /// Negative log PDF/PMF of x in row_ix, col_ix
    pub fn surprisal(
        &self,
        x: &DType,
        row_ix: usize,
        col_ix: usize,
    ) -> Option<f64> {
        if x.is_missing() {
            return None;
        }
        let logps: Vec<f64> = self.states
            .iter()
            .map(|state| {
                let view_ix = state.asgn.asgn[col_ix];
                let k = state.views[view_ix].asgn.asgn[row_ix];
                state.views[view_ix].ftrs[&col_ix].cpnt_logp(x, k)
            })
            .collect();
        let s = -logsumexp(&logps) + (self.nstates() as f64).ln();
        Some(s)
    }

    pub fn self_surprisal(&self, row_ix: usize, col_ix: usize) -> Option<f64> {
        let x = self.data.get(row_ix, col_ix);
        self.surprisal(&x, row_ix, col_ix)
    }

    pub fn get_datum(&self, row_ix: usize, col_ix: usize) -> DType {
        self.states[0].get_datum(row_ix, col_ix)
    }

    // TODO: Should take vector of vectors and compute multiple probabilities
    // to save recomputing the weights.
    pub fn logp(
        &self,
        col_ixs: &Vec<usize>,
        vals: &Vec<Vec<DType>>,
        given_opt: &Given,
    ) -> Vec<f64> {
        let logps: Vec<Vec<f64>> = self.states
            .iter()
            .map(|state| utils::state_logp(state, &col_ixs, &vals, &given_opt))
            .collect();

        let log_nstates = (self.nstates() as f64).ln();
        transpose(&logps)
            .iter()
            .map(|lps| logsumexp(&lps) - log_nstates)
            .collect()
    }

    /// Simulate values from joint or conditional distribution
    pub fn simulate(
        &self,
        col_ixs: &Vec<usize>,
        given_opt: &Option<Vec<(usize, DType)>>,
        n: usize,
        mut rng: &mut Rng,
    ) -> Vec<Vec<DType>> {
        let weights = utils::given_weights(&self.states, &col_ixs, &given_opt);
        let state_ixer = Categorical::flat(self.nstates());

        (0..n)
            .map(|_| {
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
            })
            .collect()
    }

    /// Return the most likely value for a cell in the table along with the
    /// confidence in that imputaion.
    pub fn impute(
        &self,
        row_ix: usize,
        col_ix: usize,
        with_unc: bool,
    ) -> (DType, f64) {
        let val: DType = match self.ftype(col_ix) {
            FType::Continuous => {
                let x = utils::continuous_impute(&self.states, row_ix, col_ix);
                DType::Continuous(x)
            }
            FType::Categorical => {
                let x = utils::categorical_impute(&self.states, row_ix, col_ix);
                DType::Categorical(x)
            }
        };
        let unc = if with_unc {
            utils::kl_uncertainty(&self.states, row_ix, col_ix)
        } else {
            -1.0
        };
        (val, unc)
    }

    /// Return the most likely value for a column given a set of conditions
    /// along with the confidence in that prediction.
    pub fn predict(&self, col_ix: usize, given: &Given) -> (DType, f64) {
        match self.ftype(col_ix) {
            FType::Continuous => {
                let x = utils::continuous_predict(&self.states, col_ix, &given);
                (DType::Continuous(x), 0.0)
            }
            FType::Categorical => {
                let x =
                    utils::categorical_predict(&self.states, col_ix, &given);
                (DType::Categorical(x), 0.0)
            }
        }
    }

    // TODO: Use JS Divergence?
    // TODO: Use 1 - KL and reframe as certainty?
    /// Computes the predictive uncertainty for the datum at (row_ix, col_ix)
    /// as mean the pairwise KL divergence between the components to which the
    /// datum is assigned.
    pub fn predictive_uncertainty(
        &self,
        row_ix: usize,
        col_ix: usize,
        n_samples: usize,
        mut rng: &mut Rng,
    ) -> f64 {
        if n_samples > 0 {
            utils::js_uncertainty(
                &self.states,
                row_ix,
                col_ix,
                n_samples,
                &mut rng,
            )
        } else {
            utils::kl_uncertainty(&self.states, row_ix, col_ix)
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;

    fn oracle_from_yaml(filenames: Vec<&str>) -> Oracle {
        let states = utils::load_states(filenames);
        let data = DataStore::new(states[0].clone_data());
        Oracle {
            states: states,
            codebook: Codebook::default(),
            data: data,
        }
    }

    const TOL: f64 = 1E-8;
    fn get_single_continuous_oracle_from_yaml() -> Oracle {
        let filenames = vec!["resources/test/single-continuous.yaml"];
        oracle_from_yaml(filenames)
    }

    fn get_duplicate_single_continuous_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/single-continuous.yaml",
            "resources/test/single-continuous.yaml",
        ];
        oracle_from_yaml(filenames)
    }

    fn get_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/small-state-1.yaml",
            "resources/test/small-state-2.yaml",
            "resources/test/small-state-3.yaml",
        ];

        oracle_from_yaml(filenames)
    }

    #[test]
    fn single_continuous_column_logp() {
        let oracle = get_single_continuous_oracle_from_yaml();

        let vals = vec![vec![DType::Continuous(-1.0)]];
        let logp = oracle.logp(&vec![0], &vals, &None)[0];

        assert_relative_eq!(logp, -2.7941051646651953, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_logp_duplicated_states() {
        let oracle = get_duplicate_single_continuous_oracle_from_yaml();

        let vals = vec![vec![DType::Continuous(-1.0)]];
        let logp = oracle.logp(&vec![0], &vals, &None)[0];

        assert_relative_eq!(logp, -2.7941051646651953, epsilon = TOL);
    }

    #[test]
    fn mutual_information_smoke() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let mi_01 = oracle.mi(0, 1, 1000, MiType::UnNormed, &mut rng);
        let mi_02 = oracle.mi(0, 2, 1000, MiType::UnNormed, &mut rng);
        let mi_12 = oracle.mi(1, 2, 1000, MiType::UnNormed, &mut rng);

        assert!(mi_01 > 0.0);
        assert!(mi_02 > 0.0);
        assert!(mi_12 > 0.0);
    }

    #[test]
    fn surpisal_value() {
        let oracle = get_oracle_from_yaml();
        let s = oracle.surprisal(&DType::Continuous(1.2), 3, 1).unwrap();
        assert_relative_eq!(s, 1.7739195803316758, epsilon = 10E-7);
    }

    #[test]
    fn kl_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();
        let u = oracle.predictive_uncertainty(0, 1, 0, &mut rng);

        assert!(u > 0.0);
    }
}

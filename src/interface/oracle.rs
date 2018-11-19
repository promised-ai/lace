extern crate csv;
extern crate itertools;
extern crate rand;
extern crate rmp_serde;
extern crate rusqlite;
extern crate rv;
extern crate serde_json;
extern crate serde_yaml;

use std::collections::{BTreeMap, HashSet};
use std::f64::NEG_INFINITY;
use std::io::Result;
use std::iter::FromIterator;

use self::rand::Rng;
use self::rv::dist::{Categorical, Gaussian, Mixture};
use self::rv::traits::Rv;

use cc::file_utils;
use cc::state::StateDiagnostics;
use cc::{Codebook, DataStore, Datum, FType, State};
use interface::{utils, Engine, Given};
use misc::{logsumexp, transpose};
use rayon::prelude::*;
use stats::pit::{combine_mixtures, pit};

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
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum MiType {
    /// The Standard, un-normalized variant
    UnNormed,
    /// Normalized by the max MI, which is `min(H(A), H(B))`
    Normed,
    /// Linfoot information Quantity. Derived by computing the mutual
    /// information between the two components of a bivariate Normal with
    /// covariance rho, and solving for rho.
    Linfoot,
    /// Variation of Information. A version of mutual information that
    /// satisfies the triangle inequality.
    Voi,
    /// Jaccard distance between X an Y. Jaccard(X, Y) is in [0, 1].
    Jaccard,
    /// Information Quality Ratio:  the amount of information of a variable
    /// based on another variable against total uncertainty.
    Iqr,
    /// Mutual Information normed the with square root of the product of the
    /// components entropies. Akin to the Pearson correlation coefficient.
    Pearson,
}

/// The type of uncertainty to use for `Oracle.impute`
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum ImputeUncertaintyType {
    /// Given a set of distributions Θ = {Θ<sub>1</sub>, ..., Θ<sub>n</sub>},
    /// return the mean of KL(Θ<sub>i</sub> || Θ<sub>i</sub>)
    PairwiseKl,
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    JsDivergence,
}

/// The type of uncertainty to use for `Oracle.predict`
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum PredictUncertaintyType {
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    JsDivergence,
}

impl Oracle {
    /// Convert an `Engine` into and `Oracle`
    pub fn from_engine(engine: Engine) -> Self {
        let data = {
            let data_map = engine.states.values().nth(0).unwrap().clone_data();
            DataStore::new(data_map)
        };

        // TODO: would be nice to have a draining iterator on the states
        // rather than cloning them
        let states: Vec<State> = engine
            .states
            .values()
            .map(|state| {
                let mut state_clone = state.clone();
                state_clone.drop_data();
                state_clone
            })
            .collect();

        Oracle {
            data,
            states,
            codebook: engine.codebook,
        }
    }

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
            codebook,
            data: DataStore::new(data),
        })
    }

    /// Returns the diagnostics for each state
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

    /// Return the FType of the columns `col_ix`
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
    ///
    /// # Arguments
    /// - row_a: the first row index
    /// - row_b: the second row index
    /// - wrt: an optional vector of column indices to contsrain the similarity.
    ///   Only the view to which the columns in `wrt` are assigned will be
    ///   considered in the similarity calculation
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

    /// Estimate the mutual information between `col_a` and `col_b` using Monte
    /// Carlo integration
    ///
    /// **Note**: If both columns are categorical, the mutual information will
    /// be computed exactly.
    ///
    /// # Arguments
    /// - col_a: the first column index
    /// - col_b: the second column index
    /// - n: the number of samples for the Monte Carlo integral
    /// - mi_type: the type of mutual information to return.
    pub fn mi(
        &self,
        col_a: usize,
        col_b: usize,
        n: usize,
        mi_type: MiType,
        mut rng: &mut impl Rng,
    ) -> f64 {
        let both_categorical = self.ftype(col_a) == FType::Categorical
            && self.ftype(col_b) == FType::Categorical;
        let (h_a, h_b, h_ab) = if both_categorical {
            let h_a = utils::categorical_entropy_single(col_a, &self.states);
            let h_b = utils::categorical_entropy_single(col_b, &self.states);
            let h_ab =
                utils::categorical_entropy_dual(col_a, col_b, &self.states);

            (h_a, h_b, h_ab)
        } else {
            let col_ixs = vec![col_a, col_b];

            let vals_ab = self.simulate(&col_ixs, &None, n, &mut rng);
            // FIXME: Do these have to be simulated independently
            let vals_a =
                vals_ab.iter().map(|vals| vec![vals[0].clone()]).collect();
            let vals_b =
                vals_ab.iter().map(|vals| vec![vals[1].clone()]).collect();

            let h_ab = self.entropy_from_samples(&vals_ab, &col_ixs);
            let h_a = self.entropy_from_samples(&vals_a, &vec![col_a]);
            let h_b = self.entropy_from_samples(&vals_b, &vec![col_b]);

            (h_a, h_b, h_ab)
        };

        let mi = h_a + h_b - h_ab;

        // https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants
        match mi_type {
            MiType::UnNormed => mi,
            MiType::Normed => mi / h_a.min(h_b),
            MiType::Voi => h_a + h_b - 2.0 * mi,
            MiType::Pearson => mi / (h_a * h_b).sqrt(),
            MiType::Iqr => mi / h_ab,
            MiType::Jaccard => 1.0 - mi / h_ab,
            MiType::Linfoot => (1.0 - (-2.0 * mi).exp()).sqrt(),
        }
    }

    pub fn mi_pw(
        &self,
        pairs: &Vec<(usize, usize)>,
        n: usize,
        mi_type: MiType,
        mut rng: &mut impl Rng,
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
    ///
    /// # Arguments
    /// - col_ixs: vector of column indices
    /// - n: number of samples for the Monte Carlo integral
    pub fn entropy(
        &self,
        col_ixs: &Vec<usize>,
        n: usize,
        mut rng: &mut impl Rng,
    ) -> f64 {
        let vals = self.simulate(&col_ixs, &None, n, &mut rng);
        self.entropy_from_samples(&vals, &col_ixs)
    }

    fn entropy_from_samples(
        &self,
        vals: &Vec<Vec<Datum>>,
        col_ixs: &Vec<usize>,
    ) -> f64 {
        self.logp(&col_ixs, &vals, &None)
            .iter()
            .fold(0.0, |acc, logp| acc - logp)
            / (vals.len() as f64)
    }

    /// Conditional entropy H(T|X) where X is lists of column indices
    ///
    /// # Arguments
    /// - col_t: the target column index
    /// - col_x: the observed column index
    /// - n: the number of samples for the Monte Carlo integral
    pub fn conditional_entropy(
        &self,
        col_t: usize,
        cols_x: &Vec<usize>,
        n: usize,
        mut rng: &mut impl Rng,
    ) -> f64 {
        // Monte Carlo approximation
        // https://en.wikipedia.org/wiki/Conditional_entropy#Definition
        let mut col_ixs = vec![col_t];
        col_ixs.append(&mut cols_x.clone());

        let tx_vals = self.simulate(&col_ixs, &None, n, &mut rng);
        let tx_logp = self.logp(&col_ixs, &tx_vals, &None);

        let t_vals = tx_vals.iter().map(|tx| vec![tx[0].clone()]).collect();
        let t_logp = self.logp(&vec![col_t], &t_vals, &None);

        t_logp
            .iter()
            .zip(tx_logp)
            .fold(0.0, |acc, (ft, ftx)| acc + ft - ftx)
            / (n as f64)
    }

    /// Negative log PDF/PMF of x in row_ix, col_ix.
    ///
    /// # Arguments
    /// - x: the value of which to compute the surprisal
    /// - row_ix: the hypothetical row index of `x`
    /// - col_ix: the hypothetical column index of `x`
    ///
    /// # Returns
    /// `None` if x is `Missing`, otherwise returns `Some(value)`
    pub fn surprisal(
        &self,
        x: &Datum,
        row_ix: usize,
        col_ix: usize,
    ) -> Option<f64> {
        if x.is_missing() {
            return None;
        }
        let logps: Vec<f64> = self
            .states
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

    pub fn get_datum(&self, row_ix: usize, col_ix: usize) -> Datum {
        self.data.get(row_ix, col_ix)
    }

    /// Compute the log PDF/PMF of a set of values possibly conditioned on the
    /// values of other columns
    ///
    /// # Arguments
    /// - col_ixs: An d-length vector of the indices of the columns comprising
    ///   the data.
    /// - vals: An n-length vector of d-length vectors. The joint probability of
    ///   each of the n entries will be computed.
    /// - given_opt: an optional set of observations on which to condition the
    ///   PMF/PDF
    ///
    /// # Returns
    /// A vector, `p`, where `p[i]` is the log PDF/PMF corresponding to the data
    /// in `vals[i]`.
    pub fn logp(
        &self,
        col_ixs: &Vec<usize>,
        vals: &Vec<Vec<Datum>>,
        given_opt: &Given,
    ) -> Vec<f64> {
        let logps: Vec<Vec<f64>> = self
            .states
            .iter()
            .map(|state| utils::state_logp(state, &col_ixs, &vals, &given_opt))
            .collect();

        let log_nstates = (self.nstates() as f64).ln();
        transpose(&logps)
            .iter()
            .map(|lps| logsumexp(&lps) - log_nstates)
            .collect()
    }

    /// Draw `n` samples from the cell at `[row_ix, col_ix]`.
    ///
    /// # Arguments
    /// - row_ix: the row index
    /// - col_ix, the column index
    /// - n: the optional number of draws to collect. If `None`, one draw  will
    ///   be taken.
    pub fn draw(
        &self,
        row_ix: usize,
        col_ix: usize,
        n: Option<usize>,
        mut rng: &mut impl Rng,
    ) -> Vec<Datum> {
        let state_ixer = Categorical::uniform(self.nstates());
        let n_samples: usize = n.unwrap_or(1);
        (0..n_samples)
            .map(|_| {
                // choose a random state
                let state_ix: usize = state_ixer.draw(&mut rng);
                let state = &self.states[state_ix];

                // Draw from the propoer component in the feature
                let view_ix = state.asgn.asgn[col_ix];
                let cpnt_ix = state.views[view_ix].asgn.asgn[row_ix];
                let ftr = state.get_feature(col_ix);
                ftr.draw(cpnt_ix, &mut rng)
            })
            .collect()
    }

    /// Simulate values from joint or conditional distribution
    ///
    /// # Arguments
    /// - col_ixs: a d-length vector containing the column indices to simulate
    /// - given: optional observations by which to constrain the simulation,
    ///   i.e., simulate from p(col_ixs|given)
    /// - n: the number of simulation
    ///
    /// # Returns
    /// An n-by-d vector of vectors, `x`,  where `x[i][j]` is the
    /// j<sup>th</sup> dimension of the i<sup>th</sup> simulation.
    pub fn simulate(
        &self,
        col_ixs: &Vec<usize>,
        given: &Given,
        n: usize,
        mut rng: &mut impl Rng,
    ) -> Vec<Vec<Datum>> {
        let weights = utils::given_weights(&self.states, &col_ixs, &given);
        let state_ixer = Categorical::uniform(self.nstates());

        (0..n)
            .map(|_| {
                // choose a random state
                let state_ix: usize = state_ixer.draw(&mut rng);
                let state = &self.states[state_ix];

                // for each view
                //   choose a random component from the weights
                let mut cpnt_ixs: BTreeMap<usize, usize> = BTreeMap::new();
                for (view_ix, view_weights) in &weights[state_ix] {
                    let component_ixer = {
                        let z = logsumexp(&view_weights);
                        let normed_weights: Vec<f64> =
                            view_weights.iter().map(|&w| w - z).collect();
                        Categorical::from_ln_weights(normed_weights).unwrap()
                    };
                    let k = component_ixer.draw(&mut rng);
                    cpnt_ixs.insert(*view_ix, k);
                }

                // for eacch column
                //   draw from appropriate component from that view
                let mut xs: Vec<Datum> = Vec::with_capacity(col_ixs.len());
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
    /// confidence in that imputation.
    ///
    /// # Arguments
    /// - row_ix: the row index of the cell to impute
    /// - col_ix: the column index of the cell to impute
    /// - with_unc: if `true` compute the uncertainty, otherwise a value of -1
    ///   is returned in the uncertainty spot
    ///
    /// # Returns
    /// A `(value, uncertainty_option)` tuple. If `with_unc` is `false`,
    /// `uncertainty` is -1.
    pub fn impute(
        &self,
        row_ix: usize,
        col_ix: usize,
        unc_type_opt: Option<ImputeUncertaintyType>,
    ) -> (Datum, Option<f64>) {
        let val: Datum = match self.ftype(col_ix) {
            FType::Continuous => {
                let x = utils::continuous_impute(&self.states, row_ix, col_ix);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x = utils::categorical_impute(&self.states, row_ix, col_ix);
                Datum::Categorical(x)
            }
        };
        let unc_opt = match unc_type_opt {
            Some(unc_type) => {
                Some(self.impute_uncertainty(row_ix, col_ix, unc_type))
            }
            None => None,
        };
        (val, unc_opt)
    }

    /// Return the most likely value for a column given a set of conditions
    /// along with the confidence in that prediction.
    ///
    /// # Arguments
    /// - col_ix: the index of the column to predict
    /// - given: optional observations by which to constrain the prediction
    ///
    /// # Returns
    /// A `(value, uncertainty_option)` Tuple
    ///
    /// **WARNING**: Uncertainty is not currently computed, a filler value of 0
    /// is supplied.
    pub fn predict(
        &self,
        col_ix: usize,
        given: &Given,
        unc_type_opt: Option<PredictUncertaintyType>,
    ) -> (Datum, Option<f64>) {
        let value = match self.ftype(col_ix) {
            FType::Continuous => {
                let x = utils::continuous_predict(&self.states, col_ix, &given);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x =
                    utils::categorical_predict(&self.states, col_ix, &given);
                Datum::Categorical(x)
            }
        };
        let unc_opt = match unc_type_opt {
            Some(_) => Some(self.predict_uncertainty(col_ix, &given)),
            None => None,
        };
        (value, unc_opt)
    }

    /// Computes the predictive uncertainty for the datum at (row_ix, col_ix)
    /// as mean the pairwise KL divergence between the components to which the
    /// datum is assigned.
    ///
    /// # Notes
    /// Impute uncertainty applies only to impute operations where we want to
    /// recover a specific missing (or not missing) entry. There is no special
    /// handling of non-missing entries.
    ///
    /// # Arguments
    /// - row_ix: the row index
    /// - col_ix: the column index
    /// - n_samples: the number of samples for the Monte Carlo integral. If
    ///   `n_samples` is 0, then pairwise KL divergence will be used, otherwise
    ///   JS divergence will be approximated.
    pub fn impute_uncertainty(
        &self,
        row_ix: usize,
        col_ix: usize,
        unc_type: ImputeUncertaintyType,
    ) -> f64 {
        match unc_type {
            ImputeUncertaintyType::JsDivergence => {
                utils::js_impute_uncertainty(&self.states, row_ix, col_ix)
            }
            ImputeUncertaintyType::PairwiseKl => {
                utils::kl_impute_uncertainty(&self.states, row_ix, col_ix)
            }
        }
    }

    /// Computes the uncertainty associated with predicting the value of a
    /// features with optional given conditions. Uses Jensen-Shannon divergence
    /// computed on the mixture of mixtures.
    ///
    /// # Notes
    /// Predict uncertainty applies only to prediction of hypothetical values,
    /// and not to imputation of in-table values.
    ///
    /// # Arguments
    /// - col_ix: the column index
    /// - given_opt: an optional list of (column index, value) tuples
    ///   designating other observations on which to condition the prediciton
    pub fn predict_uncertainty(
        &self,
        col_ix: usize,
        given_opt: &Option<Vec<(usize, Datum)>>,
    ) -> f64 {
        utils::predict_uncertainty(&self.states, col_ix, given_opt)
    }

    /// Compute the Probability Integral Transform (PIT) of the column at
    /// `col_ix`.
    ///
    /// # Returns
    /// An `(error, centroid)` tuple where error is the area between the 1:1
    /// correspondence line and the PIT CDF, and the centroid is the centroid of
    /// the error.
    pub fn pit(&self, col_ix: usize) -> (f64, f64) {
        // extract the feature from the first state
        let ftr = self.states[0].get_feature(col_ix);
        // TODO: can this replicated code be macroed away?
        if ftr.is_continuous() {
            let mixtures: Vec<Mixture<Gaussian>> = self
                .states
                .iter()
                .map(|state| {
                    state.get_feature_as_mixture(col_ix).unwrap_gaussian()
                })
                .collect();
            let mixture = combine_mixtures(&mixtures);
            let xs: Vec<f64> = (0..self.nrows())
                .filter_map(|row_ix| self.data.get(row_ix, col_ix).as_f64())
                .collect();
            pit(&xs, &mixture)
        } else if ftr.is_categorical() {
            let mixtures: Vec<Mixture<Categorical>> = self
                .states
                .iter()
                .map(|state| {
                    state.get_feature_as_mixture(col_ix).unwrap_categorical()
                })
                .collect();
            let mixture = combine_mixtures(&mixtures);
            let xs: Vec<u8> = (0..self.nrows())
                .filter_map(|row_ix| self.data.get(row_ix, col_ix).as_u8())
                .collect();
            pit(&xs, &mixture)
        } else {
            panic!("Unsupported feature type");
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
            states,
            codebook: Codebook::default(),
            data,
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

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp = oracle.logp(&vec![0], &vals, &None)[0];

        assert_relative_eq!(logp, -2.7941051646651953, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_logp_duplicated_states() {
        let oracle = get_duplicate_single_continuous_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
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
    fn surpisal_value_1() {
        let oracle = get_oracle_from_yaml();
        let s = oracle.surprisal(&Datum::Continuous(1.2), 3, 1).unwrap();
        assert_relative_eq!(s, 1.7739195803316758, epsilon = 10E-7);
    }

    #[test]
    fn surpisal_value_2() {
        let oracle = get_oracle_from_yaml();
        let s = oracle.surprisal(&Datum::Continuous(0.1), 1, 0).unwrap();
        assert_relative_eq!(s, 0.62084325305231269, epsilon = 10E-7);
    }

    #[test]
    fn kl_impute_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();
        let u =
            oracle.impute_uncertainty(0, 1, ImputeUncertaintyType::PairwiseKl);
        assert!(u > 0.0);
    }

    #[test]
    fn js_impute_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();
        let u = oracle.impute_uncertainty(
            0,
            1,
            ImputeUncertaintyType::JsDivergence,
        );
        assert!(u > 0.0);
    }

    #[test]
    fn predict_uncertainty_smoke_no_given() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();
        let u = oracle.predict_uncertainty(0, &None);
        assert!(u > 0.0);
    }

    #[test]
    fn predict_uncertainty_smoke_with_given() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();
        let given = vec![(1, Datum::Continuous(2.5))];
        let u = oracle.predict_uncertainty(0, &Some(given));
        assert!(u > 0.0);
    }

    #[test]
    #[ignore]
    fn predict_uncertainty_calipers() {
        use std::f64::NEG_INFINITY;
        let oracle = Oracle::load("resources/test/calipers.braid").unwrap();
        let xs = vec![1.0, 2.0, 2.5, 3.0, 3.5, 3.75];
        let (_, uncertainty_increasing) =
            xs.iter().fold((NEG_INFINITY, true), |acc, x| {
                let given = vec![(0, Datum::Continuous(*x))];
                let unc = oracle.predict_uncertainty(1, &Some(given));
                println!("Unc y|x={} is {}", x, unc);
                if unc > acc.0 && acc.1 {
                    (unc, true)
                } else {
                    (unc, false)
                }
            });
        assert!(uncertainty_increasing);
    }
}

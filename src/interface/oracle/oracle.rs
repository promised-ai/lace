use std::collections::{BTreeMap, HashSet};
use std::iter::FromIterator;
use std::path::Path;

use braid_codebook::codebook::Codebook;
use braid_stats::{Datum, SampleError};
use braid_utils::misc::{logsumexp, transpose};
use rand::Rng;
use rayon::prelude::*;
use rv::dist::{Categorical, Gaussian, Mixture};
use rv::traits::Rv;
use serde::{Deserialize, Serialize};

use super::error::{self, IndexError};
use super::utils;
use super::validation::{find_given_errors, find_value_conflicts};
use crate::cc::state::StateDiagnostics;
use crate::cc::{
    file_utils, DataStore, FType, Feature, State, SummaryStatistics,
};
use crate::interface::oracle::error::SurprisalError;
use crate::interface::{Engine, Given};

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
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
)]
pub enum MiType {
    /// The Standard, un-normalized variant
    #[serde(rename = "unnormed")]
    UnNormed,
    /// Normalized by the max MI, which is `min(H(A), H(B))`
    #[serde(rename = "normed")]
    Normed,
    /// Linfoot information Quantity. Derived by computing the mutual
    /// information between the two components of a bivariate Normal with
    /// covariance rho, and solving for rho.
    #[serde(rename = "linfoot")]
    Linfoot,
    /// Variation of Information. A version of mutual information that
    /// satisfies the triangle inequality.
    #[serde(rename = "voi")]
    Voi,
    /// Jaccard distance between X an Y. Jaccard(X, Y) is in [0, 1].
    #[serde(rename = "jaccard")]
    Jaccard,
    /// Information Quality Ratio:  the amount of information of a variable
    /// based on another variable against total uncertainty.
    #[serde(rename = "iqr")]
    Iqr,
    /// Mutual Information normed the with square root of the product of the
    /// components entropies. Akin to the Pearson correlation coefficient.
    #[serde(rename = "pearson")]
    Pearson,
}

/// Holds the components required to compute mutual information
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub struct MiComponents {
    /// The entropy of column a, H(A)
    pub h_a: f64,
    /// The entropy of column b, H(B)
    pub h_b: f64,
    /// The joint entropy of columns a and b, H(A, B)
    pub h_ab: f64,
}

impl MiComponents {
    #[inline]
    pub fn compute(&self, mi_type: MiType) -> f64 {
        let mi = self.h_a + self.h_b - self.h_ab;

        match mi_type {
            MiType::UnNormed => mi,
            MiType::Normed => mi / self.h_a.min(self.h_b),
            MiType::Voi => self.h_a + self.h_b - 2.0 * mi,
            MiType::Pearson => mi / (self.h_a * self.h_b).sqrt(),
            MiType::Iqr => mi / self.h_ab,
            MiType::Jaccard => 1.0 - mi / self.h_ab,
            MiType::Linfoot => (1.0 - (-2.0 * mi).exp()).sqrt(),
        }
    }
}

/// The type of uncertainty to use for `Oracle.impute`
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
)]
pub enum ImputeUncertaintyType {
    /// Given a set of distributions Θ = {Θ<sub>1</sub>, ..., Θ<sub>n</sub>},
    /// return the mean of KL(Θ<sub>i</sub> || Θ<sub>i</sub>)
    #[serde(rename = "pairwise_kl")]
    PairwiseKl,
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    #[serde(rename = "js_divergence")]
    JsDivergence,
}

/// The type of uncertainty to use for `Oracle.predict`
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum PredictUncertaintyType {
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    #[serde(rename = "js_divergence")]
    JsDivergence,
}

//
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
)]
pub enum ConditionalEntropyType {
    /// Normal conditional entropy
    UnNormed,
    /// IP(X; Y), The proportion of information in X accounted for by Y
    InfoProp,
}

impl Oracle {
    /// Convert an `Engine` into an `Oracle`
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
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let config = file_utils::load_file_config(dir).unwrap_or_default();
        let data = file_utils::load_data(dir, &config)?;
        let mut states = file_utils::load_states(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;

        // Move states from map to vec
        let ids: Vec<usize> = states.keys().copied().collect();
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
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// assert_eq!(oracle.nstates(), 8);
    /// ```
    pub fn nstates(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of rows in the `Oracle`
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// assert_eq!(oracle.nrows(), 50);
    /// ```
    pub fn nrows(&self) -> usize {
        self.states[0].nrows()
    }

    /// Returns the number of columns/features in the `Oracle`
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// assert_eq!(oracle.ncols(), 85);
    /// ```
    pub fn ncols(&self) -> usize {
        self.states[0].ncols()
    }

    /// Return the FType of the column `col_ix`
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::cc::FType;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let ftype = oracle.ftype(Column::Swims.into()).unwrap();
    ///
    /// assert_eq!(ftype, FType::Categorical);
    /// ```
    pub fn ftype(&self, col_ix: usize) -> Result<FType, IndexError> {
        if col_ix < self.ncols() {
            let state = &self.states[0];
            let view_ix = state.asgn.asgn[col_ix];
            Ok(state.views[view_ix].ftrs[&col_ix].ftype())
        } else {
            Err(IndexError::ColumnIndexOutOfBoundsError)
        }
    }

    /// Returns a vector of the feature types of each row
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let ftypes = oracle.ftypes();
    ///
    /// assert!(ftypes.iter().all(|ftype| ftype.is_categorical()));
    /// ```
    pub fn ftypes(&self) -> Vec<FType> {
        (0..self.ncols())
            .map(|col_ix| self.ftype(col_ix).unwrap())
            .collect()
    }

    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::Column;
    /// use braid::cc::SummaryStatistics;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let swims_summary = oracle.summarize_col(Column::Swims.into()).unwrap();
    ///
    /// match swims_summary {
    ///     SummaryStatistics::Categorical { min, max, mode } => {
    ///         assert_eq!(min, 0);
    ///         assert_eq!(max, 1);
    ///         assert_eq!(mode, vec![0]);
    ///     }
    ///     _ => panic!("should be categorical")
    /// }
    /// ```
    pub fn summarize_col(
        &self,
        col_ix: usize,
    ) -> Result<SummaryStatistics, IndexError> {
        if col_ix < self.ncols() {
            Ok(self.data.summarize_col(col_ix))
        } else {
            Err(IndexError::ColumnIndexOutOfBoundsError)
        }
    }

    /// Estimated dependence probability between `col_a` and `col_b`
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let depprob_flippers = oracle.depprob(
    ///     Column::Swims.into(),
    ///     Column::Flippers.into()
    /// ).unwrap();
    ///
    /// let depprob_fast = oracle.depprob(
    ///     Column::Swims.into(),
    ///     Column::Fast.into()
    /// ).unwrap();
    ///
    /// assert!(depprob_flippers > depprob_fast);
    /// ```
    pub fn depprob(
        &self,
        col_a: usize,
        col_b: usize,
    ) -> Result<f64, IndexError> {
        let ncols = self.ncols();
        if col_a < ncols && col_b < ncols {
            let depprob = self.states.iter().fold(0.0, |acc, state| {
                if state.asgn.asgn[col_a] == state.asgn.asgn[col_b] {
                    acc + 1.0
                } else {
                    acc
                }
            }) / (self.nstates() as f64);
            Ok(depprob)
        } else {
            Err(IndexError::ColumnIndexOutOfBoundsError)
        }
    }

    /// Compute dependence probability for a list of column pairs.
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let depprobs = oracle.depprob_pw(&vec![(1, 12), (3, 2)]).unwrap();
    ///
    /// assert_eq!(depprobs.len(), 2);
    /// assert_eq!(depprobs[0], oracle.depprob(1, 12).unwrap());
    /// assert_eq!(depprobs[1], oracle.depprob(3, 2).unwrap());
    /// ```
    pub fn depprob_pw(
        &self,
        pairs: &[(usize, usize)],
    ) -> Result<Vec<f64>, IndexError> {
        if pairs.len() == 0 {
            return Ok(Vec::new());
        }
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
    /// - wrt: an optional vector of column indices to constrain the similarity.
    ///   Only the view to which the columns in `wrt` are assigned will be
    ///   considered in the similarity calculation
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::Row;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let rowsim = oracle.rowsim(
    ///     Row::Wolf.into(),
    ///     Row::Collie.into(),
    ///     None
    /// ).unwrap();
    ///
    /// assert!(rowsim >= 0.0 && rowsim <= 1.0);
    /// ```
    /// Adding context with `wrt` (with respect to):
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::examples::animals::Row;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// # let rowsim =oracle.rowsim(
    /// #     Row::Wolf.into(),
    /// #     Row::Collie.into(),
    /// # None).unwrap();
    /// use braid::examples::animals::Column;
    ///
    /// let rowsim_wrt = oracle.rowsim(
    ///     Row::Wolf.into(),
    ///     Row::Collie.into(),
    ///     Some(&vec![Column::Swims.into()])
    /// ).unwrap();
    ///
    /// assert_ne!(rowsim, rowsim_wrt);
    /// ```
    pub fn rowsim(
        &self,
        row_a: usize,
        row_b: usize,
        wrt: Option<&Vec<usize>>,
    ) -> Result<f64, error::RowSimError> {
        let nrows = self.nrows();
        if row_a >= nrows || row_b >= nrows {
            return Err(error::RowSimError::RowIndexOutOfBoundsError);
        }

        if let Some(col_ixs) = wrt {
            if col_ixs.is_empty() {
                return Err(error::RowSimError::EmptyWrtError);
            } else if col_ixs.iter().any(|&ix| ix >= self.ncols()) {
                return Err(error::RowSimError::WrtColumnIndexOutOfBoundsError);
            }
        }

        let rowsim = self.states.iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt {
                Some(col_ixs) => {
                    let asgn = &state.asgn.asgn;
                    let viewset: HashSet<usize> = HashSet::from_iter(
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]),
                    );
                    viewset.iter().copied().collect()
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
        }) / self.nstates() as f64;

        Ok(rowsim)
    }

    /// Compute row similarity for pairs of rows
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::Row;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let rowsims = oracle.rowsim_pw(
    ///     &vec![
    ///         (Row::Gorilla.into(), Row::SpiderMonkey.into()),
    ///         (Row::Gorilla.into(), Row::Skunk.into()),
    ///     ],
    ///     None
    /// ).unwrap();
    ///
    /// assert!(rowsims.iter().all(|&rowsim| 0.0 <= rowsim && rowsim <= 1.0));
    /// ```
    pub fn rowsim_pw(
        &self,
        pairs: &[(usize, usize)],
        wrt: Option<&Vec<usize>>,
    ) -> Result<Vec<f64>, error::RowSimError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        // TODO: Speed up by recomputing the view indices for each state
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
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::interface::MiType;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mi_flippers = oracle.mi(
    ///     Column::Swims.into(),
    ///     Column::Flippers.into(),
    ///     1000,
    ///     MiType::Iqr,
    /// ).unwrap();
    ///
    /// let mi_fast = oracle.mi(
    ///     Column::Swims.into(),
    ///     Column::Fast.into(),
    ///     1000,
    ///     MiType::Iqr,
    /// ).unwrap();
    ///
    /// assert!(mi_flippers > mi_fast);
    /// ```
    ///
    /// The IQR normalized variant is normalized between 0 and 1
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::interface::MiType;
    /// # use braid::examples::animals::Column;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// let mi_self = oracle.mi(
    ///     Column::Swims.into(),
    ///     Column::Swims.into(),
    ///     1000,
    ///     MiType::Iqr,
    /// ).unwrap();
    ///
    /// assert_eq!(mi_self, 1.0);
    /// ```
    pub fn mi(
        &self,
        col_a: usize,
        col_b: usize,
        n: usize,
        mi_type: MiType,
    ) -> Result<f64, error::MiError> {
        let ncols = self.ncols();
        if n == 0 {
            return Err(error::MiError::NIsZeroError);
        } else if col_a >= ncols || col_b >= ncols {
            return Err(error::MiError::ColumnIndexOutOfBoundsError);
        }

        let mi_cpnts = self.mi_components(col_a, col_b, n);
        Ok(mi_cpnts.compute(mi_type))
    }

    /// Compute mutual information over pairs of columns
    ///
    /// # Notes
    ///
    /// This function has special optimizations over computing oracle::mi for
    /// pairs manually.
    pub fn mi_pw(
        &self,
        pairs: &[(usize, usize)],
        n: usize,
        mi_type: MiType,
    ) -> Result<Vec<f64>, error::MiError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Precompute the single-column entropies
        let mut col_ixs: HashSet<usize> = HashSet::new();
        pairs.iter().for_each(|(col_a, col_b)| {
            col_ixs.insert(*col_a);
            col_ixs.insert(*col_b);
        });

        let ncols = self.ncols();
        for ix in col_ixs.iter() {
            if *ix >= ncols {
                return Err(error::MiError::ColumnIndexOutOfBoundsError);
            }
        }

        let mut entropies: BTreeMap<usize, f64> = BTreeMap::new();

        col_ixs.iter().for_each(|&col_ix| {
            let h = utils::entropy_single(col_ix, &self.states);
            entropies.insert(col_ix, h);
        });

        // TODO: Parallelize
        let mis: Vec<_> = pairs
            .iter()
            .map(|(col_a, col_b)| {
                let h_a = entropies[col_a];
                let h_b = entropies[col_b];
                let h_ab = self.dual_entropy(*col_a, *col_b, n);
                let mi_cpnts = MiComponents { h_a, h_b, h_ab };
                mi_cpnts.compute(mi_type)
            })
            .collect();

        Ok(mis)
    }

    /// Estimate entropy using Quasi Monte Carlo integration
    ///
    /// # Notes
    /// The exact value is provided if there is only one index in col_ixs and
    /// that column is categorical, gaussian, or labeler.
    ///
    /// # Arguments
    /// - col_ixs: vector of column indices
    /// - n: number of samples for the Quasi Monte Carlo integral.
    ///
    /// # Examples
    ///
    /// There is more information in the swims column than in the blue column
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::interface::MiType;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// // Close to uniformly distributed -> high entropy
    /// let h_swims = oracle.entropy(
    ///     &vec![Column::Swims.into()],
    ///     10_000,
    /// );
    ///
    /// // Close to deterministic -> low entropy
    /// let h_blue = oracle.entropy(
    ///     &vec![Column::Blue.into()],
    ///     10_000,
    /// );
    ///
    /// assert!(h_blue < h_swims);
    /// ```
    ///
    /// The `n` argument isn't required for a single categorical column
    /// because the exact computation is used.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::interface::MiType;
    /// # use braid::examples::animals::Column;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// let h_swims_10k = oracle.entropy(
    ///     &vec![Column::Swims.into()],
    ///     10_000,
    /// ).unwrap();
    ///
    /// let h_swims_0 = oracle.entropy(
    ///     &vec![Column::Swims.into()],
    ///     0,
    /// ).unwrap();
    ///
    /// assert!((h_swims_10k - h_swims_0).abs() < 1E-12);
    /// ```
    pub fn entropy(
        &self,
        col_ixs: &[usize],
        n: usize,
    ) -> Result<f64, error::EntropyError> {
        let ncols = self.ncols();
        if col_ixs.is_empty() {
            return Err(error::EntropyError::NoTargetColumnsError);
        } else if col_ixs.iter().any(|&ix| ix >= ncols) {
            return Err(error::EntropyError::ColumnIndexOutOfBoundsError);
        };
        Ok(self.entropy_unchecked(&col_ixs, n))
    }

    /// Compute the proportion of information in `cols_t` accounted for by
    /// `cols_x`.
    ///
    /// # Arguments
    /// - cols_t: The target columns. Typically the target of a prediction.
    /// - cols_x: The predictor columns.
    /// - n: the number of samples for the Quasi Monte Carlo integral. Make n
    ///   high enough to integrate a function with as many dimensions as there
    ///   are total columns in `cols_t` and `cols_x`.
    ///
    /// # Notes
    /// If all variables are discrete, the information proportion should be in
    /// [0, 1] (with minor deviations due to approximation error); the behavior
    /// is less predictable when one or more of the variables are continuous
    /// because entropy is a murky concept in continuous space.
    ///
    /// # Example
    ///
    /// Flippers tells us more about swimming that an animal's being fast.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let ip_flippers = oracle.info_prop(
    ///     &vec![Column::Swims.into()],
    ///     &vec![Column::Flippers.into()],
    ///     1000,
    /// ).unwrap();
    ///
    /// let ip_fast = oracle.info_prop(
    ///     &vec![Column::Swims.into()],
    ///     &vec![Column::Fast.into()],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(ip_flippers > ip_fast);
    ///
    /// assert!(ip_flippers >= 0.0);
    /// assert!(ip_flippers <= 1.0);
    ///
    /// assert!(ip_fast >= 0.0);
    /// assert!(ip_fast <= 1.0);
    /// ```
    ///
    /// Adding more predictor columns increases the information proportion
    /// monotonically.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::examples::animals::Column;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// # let ip_flippers = oracle.info_prop(
    /// #     &vec![Column::Swims.into()],
    /// #     &vec![Column::Flippers.into()],
    /// #     1000,
    /// # ).unwrap();
    /// let ip_flippers_coastal = oracle.info_prop(
    ///     &vec![Column::Swims.into()],
    ///     &vec![Column::Flippers.into(), Column::Coastal.into()],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(ip_flippers < ip_flippers_coastal);
    /// assert!(ip_flippers_coastal <= 1.0);
    ///
    /// let ip_flippers_coastal_fast = oracle.info_prop(
    ///     &vec![Column::Swims.into()],
    ///     &vec![
    ///         Column::Flippers.into(),
    ///         Column::Coastal.into(),
    ///         Column::Fast.into(),
    ///     ],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(ip_flippers_coastal < ip_flippers_coastal_fast);
    /// assert!(ip_flippers_coastal_fast <= 1.0);
    /// ```
    pub fn info_prop(
        &self,
        cols_t: &Vec<usize>,
        cols_x: &Vec<usize>,
        n: usize,
    ) -> Result<f64, error::InfoPropError> {
        let ncols = self.ncols();
        if n == 0 {
            return Err(error::InfoPropError::NIsZeroError);
        } else if cols_t.is_empty() {
            return Err(error::InfoPropError::NoTargetColumnsError);
        } else if cols_x.is_empty() {
            return Err(error::InfoPropError::NoTargetColumnsError);
        } else if cols_t.iter().any(|&ix| ix >= ncols) {
            return Err(
                error::InfoPropError::TargetColumnIndexOutOfBoundsError,
            );
        } else if cols_x.iter().any(|&ix| ix >= ncols) {
            return Err(
                error::InfoPropError::PredictorColumnIndexOutOfBoundsError,
            );
        }

        let all_cols: Vec<usize> = {
            let mut set: HashSet<usize> = HashSet::new();
            cols_t.iter().for_each(|&col| {
                set.insert(col);
            });
            cols_x.iter().for_each(|&col| {
                set.insert(col);
            });
            set.drain().collect()
        };

        // The target column is among the predictors, which means that all the
        // information is recovered.
        if all_cols.len() != cols_x.len() + cols_t.len() {
            return Ok(1.0);
        }

        let h_all = self.entropy_unchecked(&all_cols, n);
        let h_t = self.entropy_unchecked(&cols_t, n);
        let h_x = self.entropy_unchecked(&cols_x, n);

        Ok((h_t + h_x - h_all) / h_t)
    }

    /// Conditional entropy H(T|X) where X is lists of column indices
    ///
    /// # Arguments
    /// - col_t: the target column index
    /// - col_x: the observed column index
    /// - n: the number of samples for the Monte Carlo integral
    ///
    /// # Example
    /// Knowing whether something has flippers leaves less information to
    /// account for WRT its swimming than does knowing whether it is fast and
    /// has a tail.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mi_flippers = oracle.conditional_entropy(
    ///     Column::Swims.into(),
    ///     &vec![Column::Flippers.into()],
    ///     1000,
    /// ).unwrap();
    ///
    /// let mi_fast_tail = oracle.conditional_entropy(
    ///     Column::Swims.into(),
    ///     &vec![Column::Fast.into(), Column::Tail.into()],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(mi_flippers < mi_fast_tail);
    /// ```
    pub fn conditional_entropy(
        &self,
        col_t: usize,
        cols_x: &[usize],
        n: usize,
    ) -> Result<f64, error::ConditionalEntropyError> {
        let ncols = self.ncols();
        if n == 0 {
            return Err(error::ConditionalEntropyError::NIsZeroError);
        } else if cols_x.is_empty() {
            return Err(
                error::ConditionalEntropyError::NoPredictorColumnsError,
            );
        } else if col_t >= ncols {
            return Err(error::ConditionalEntropyError::TargetColumnIndexOutOfBoundsError);
        } else if cols_x.iter().any(|&ix| ix >= ncols) {
            return Err(error::ConditionalEntropyError::PredictorColumnIndexOutOfBoundsError);
        }

        // The target is a predictor, which means there is no left over entropy
        if cols_x.iter().find(|&&ix| ix == col_t).is_some() {
            return Ok(0.0);
        }

        let all_cols: Vec<_> = {
            let mut col_ixs: HashSet<usize> = HashSet::new();
            col_ixs.insert(col_t);
            cols_x.iter().for_each(|&ix| {
                col_ixs.insert(ix);
            });
            col_ixs.drain().collect()
        };

        // make sure that there were no duplicate columns anywhere
        if all_cols.len() < cols_x.len() {
            return Err(
                error::ConditionalEntropyError::DuplicatePredictorsError,
            );
        }

        let h_x = self.entropy_unchecked(cols_x, n);
        let h_all = self.entropy_unchecked(&all_cols, n);

        Ok(h_all - h_x)
    }

    /// Pairwise copmutation of conditional entreopy or information proportion
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::interface::ConditionalEntropyType;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let col_pairs: Vec<(usize, usize)> = vec![
    ///     (Column::Swims.into(), Column::Flippers.into()),
    ///     (Column::Swims.into(), Column::Fast.into()),
    /// ];
    ///
    /// let ce = oracle.conditional_entropy_pw(
    ///     &col_pairs,
    ///     1000,
    ///     ConditionalEntropyType::UnNormed
    /// ).unwrap();
    ///
    /// assert_eq!(ce.len(), 2);
    /// assert!(ce[0] < ce[1]);
    /// ```
    ///
    /// ... and specify information proportion instead of un-normalized
    /// conditional entropy changes the relationships.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::examples::animals::Column;
    /// # use braid::interface::ConditionalEntropyType;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// # let col_pairs: Vec<(usize, usize)> = vec![
    /// #     (Column::Swims.into(), Column::Flippers.into()),
    /// #     (Column::Swims.into(), Column::Fast.into()),
    /// # ];
    /// let info_prop = oracle.conditional_entropy_pw(
    ///     &col_pairs,
    ///     1000,
    ///     ConditionalEntropyType::InfoProp
    /// ).unwrap();
    ///
    /// assert_eq!(info_prop.len(), 2);
    /// assert!(info_prop[0] > info_prop[1]);
    /// ```
    pub fn conditional_entropy_pw(
        &self,
        col_pairs: &[(usize, usize)],
        n: usize,
        kind: ConditionalEntropyType,
    ) -> Result<Vec<f64>, error::ConditionalEntropyError> {
        if col_pairs.is_empty() {
            return Ok(vec![]);
        } else if n == 0 {
            return Err(error::ConditionalEntropyError::NIsZeroError);
        };

        let ncols = &self.ncols();

        col_pairs
            .par_iter()
            .map(|(col_a, col_b)| {
                if col_a >= ncols {
                    Err(error::ConditionalEntropyError::TargetColumnIndexOutOfBoundsError)
                } else if col_b >= ncols {
                    Err(error::ConditionalEntropyError::PredictorColumnIndexOutOfBoundsError)
                } else {
                    match kind {
                        ConditionalEntropyType::InfoProp => {
                            let MiComponents { h_a, h_b, h_ab } =
                                self.mi_components(*col_a, *col_b, n);
                            Ok((h_a + h_b - h_ab) / h_a)
                        }
                        ConditionalEntropyType::UnNormed => {
                            let h_b = utils::entropy_single(*col_b, &self.states);
                            let h_ab = self.dual_entropy(*col_a, *col_b, n);
                            Ok(h_ab - h_b)
                        }
                    }
                }
            })
            .collect()
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
    ///
    /// # Example
    ///
    /// A pig being fierce is more surprising than a lion being fierce.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid_stats::Datum;
    /// use braid::examples::animals::{Column, Row};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let present = Datum::Categorical(1);
    ///
    /// let s_pig = oracle.surprisal(
    ///     &present,
    ///     Row::Pig.into(),
    ///     Column::Fierce.into()
    /// ).unwrap();
    ///
    /// let s_lion = oracle.surprisal(
    ///     &present,
    ///     Row::Lion.into(),
    ///     Column::Fierce.into()
    /// ).unwrap();
    ///
    /// assert!(s_pig > s_lion);
    /// ```
    pub fn surprisal(
        &self,
        x: &Datum,
        row_ix: usize,
        col_ix: usize,
    ) -> Result<Option<f64>, error::SurprisalError> {
        let ftype_ok: bool = self
            .ftype(col_ix)
            .map_err(|_| error::SurprisalError::ColumnIndexOutOfBoundsError)
            .map(|ftype| ftype.datum_compatible(x))?;

        if !ftype_ok {
            return Err(error::SurprisalError::InvalidDatumForColumnError);
        } else if row_ix >= self.nrows() {
            return Err(error::SurprisalError::RowIndexOutOfBoundsError);
        } else if col_ix >= self.ncols() {
            return Err(error::SurprisalError::ColumnIndexOutOfBoundsError);
        }

        Ok(self.surprisal_unchecked(x, row_ix, col_ix))
    }

    /// Get the surprisal of the datum in a cell.
    ///
    /// # Example
    ///
    /// A pig is fierce, which is more surprising than a lion being fierce.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::{Column, Row};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let s_pig = oracle.self_surprisal(
    ///     Row::Pig.into(),
    ///     Column::Fierce.into()
    /// ).unwrap();
    ///
    /// let s_lion = oracle.self_surprisal(
    ///     Row::Lion.into(),
    ///     Column::Fierce.into()
    /// ).unwrap();
    ///
    /// assert!(s_pig > s_lion);
    /// ```
    pub fn self_surprisal(
        &self,
        row_ix: usize,
        col_ix: usize,
    ) -> Result<Option<f64>, error::SurprisalError> {
        self.datum(row_ix, col_ix)
            .map_err(|err| SurprisalError::from(err))
            .map(|x| self.surprisal_unchecked(&x, row_ix, col_ix))
    }

    /// Get the datum at an index
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid_stats::Datum;
    /// use braid::examples::animals::{Column, Row};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let x = oracle.datum(
    ///     Row::Pig.into(),
    ///     Column::Fierce.into()
    /// ).unwrap();
    ///
    /// assert_eq!(x, Datum::Categorical(1));
    /// ```
    pub fn datum(
        &self,
        row_ix: usize,
        col_ix: usize,
    ) -> Result<Datum, IndexError> {
        if row_ix >= self.nrows() {
            Err(IndexError::RowIndexOutOfBoundsError)
        } else if col_ix >= self.ncols() {
            Err(IndexError::ColumnIndexOutOfBoundsError)
        } else {
            Ok(self.data.get(row_ix, col_ix))
        }
    }

    /// Compute the log PDF/PMF of a set of values possibly conditioned on the
    /// values of other columns
    ///
    /// # Arguments
    ///
    /// - col_ixs: An d-length vector of the indices of the columns comprising
    ///   the data.
    /// - vals: An n-length vector of d-length vectors. The joint probability of
    ///   each of the n entries will be computed.
    /// - given: an optional set of observations on which to condition the
    ///   PMF/PDF
    /// - state_ixs_opt: An optional vector of the state indices to use for the
    ///   logp computation. If `None`, all states are used.
    ///
    /// # Returns
    ///
    /// A vector, `p`, where `p[i]` is the log PDF/PMF corresponding to the data
    /// in `vals[i]`.
    ///
    /// # Example
    ///
    /// The probability that an animals swims is lower than the probability
    /// that it swims given that is has flippers.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid_stats::Datum;
    /// use braid::Given;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let logp_swims = oracle.logp(
    ///     &vec![Column::Swims.into()],
    ///     &vec![vec![Datum::Categorical(1)]],
    ///     &Given::Nothing,
    ///     None,
    /// ).unwrap();
    ///
    /// let logp_swims_given_flippers = oracle.logp(
    ///     &vec![Column::Swims.into()],
    ///     &vec![vec![Datum::Categorical(1)]],
    ///     &Given::Conditions(
    ///         vec![(Column::Flippers.into(), Datum::Categorical(1))]
    ///     ),
    ///     None,
    /// ).unwrap();
    ///
    /// assert!(logp_swims[0] < logp_swims_given_flippers[0]);
    /// ```
    #[allow(clippy::ptr_arg)]
    pub fn logp(
        &self,
        col_ixs: &[usize],
        vals: &Vec<Vec<Datum>>,
        given: &Given,
        states_ixs_opt: Option<Vec<usize>>,
    ) -> Result<Vec<f64>, error::LogpError> {
        if col_ixs.is_empty() {
            Err(error::LogpError::NoTargetsError)
        } else if col_ixs.iter().any(|&ix| ix >= self.ncols()) {
            Err(error::LogpError::TargetIndexOutOfBoundsError)
        } else {
            find_given_errors(col_ixs, &self, given)
                .map_err(|err| err.into())
                .and_then(|_| find_value_conflicts(col_ixs, vals, &self))
        }?;
        match states_ixs_opt {
            Some(ref state_ixs) => {
                if state_ixs.iter().any(|&ix| ix >= self.nstates()) {
                    Err(error::LogpError::StateIndexOutOfBoundsError)
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }?;
        Ok(self.logp_unchecked(col_ixs, vals, given, states_ixs_opt))
    }

    /// Draw `n` samples from the cell at `[row_ix, col_ix]`.
    ///
    /// # Arguments
    ///
    /// - row_ix: the row index
    /// - col_ix, the column index
    /// - n: the number of draws to collect
    ///
    /// # Example
    ///
    /// Draw 12 values of a Pig's fierceness.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::{Column, Row};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mut rng = rand::thread_rng();
    /// let xs = oracle.draw(
    ///     Row::Pig.into(),
    ///     Column::Fierce.into(),
    ///     12,
    ///     &mut rng,
    /// ).unwrap();
    ///
    /// assert_eq!(xs.len(), 12);
    /// assert!(xs.iter().all(|x| x.is_categorical()));
    /// ```
    pub fn draw(
        &self,
        row_ix: usize,
        col_ix: usize,
        n: usize,
        mut rng: &mut impl Rng,
    ) -> Result<Vec<Datum>, IndexError> {
        if row_ix >= self.nrows() {
            return Err(IndexError::RowIndexOutOfBoundsError);
        } else if col_ix >= self.ncols() {
            return Err(IndexError::ColumnIndexOutOfBoundsError);
        } else if n == 0 {
            return Ok(Vec::new());
        }
        let state_ixer = Categorical::uniform(self.nstates());
        let draws: Vec<_> = (0..n)
            .map(|_| {
                // choose a random state
                let state_ix: usize = state_ixer.draw(&mut rng);
                let state = &self.states[state_ix];

                // Draw from the propoer component in the feature
                let view_ix = state.asgn.asgn[col_ix];
                let cpnt_ix = state.views[view_ix].asgn.asgn[row_ix];
                let ftr = state.feature(col_ix);
                ftr.draw(cpnt_ix, &mut rng)
            })
            .collect();
        Ok(draws)
    }

    /// Simulate values from joint or conditional distribution
    ///
    /// # Arguments
    ///
    /// - col_ixs: a d-length vector containing the column indices to simulate
    /// - given: optional observations by which to constrain the simulation,
    ///   i.e., simulate from p(col_ixs|given)
    /// - n: the number of simulation
    /// - states_ixs_opt: The indices of the states from which to simulate. If
    ///   `None`, simulate from all states.
    ///
    /// # Returns
    ///
    /// An n-by-d vector of vectors, `x`,  where `x[i][j]` is the
    /// j<sup>th</sup> dimension of the i<sup>th</sup> simulation.
    ///
    /// # Example
    ///
    /// Simulate the appearance of a hypothetical animal that is fierce and
    /// fast.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::Given;
    /// use braid::examples::animals::Column;
    /// use braid_stats::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let given = Given::Conditions(
    ///     vec![
    ///         (Column::Fierce.into(), Datum::Categorical(1)),
    ///         (Column::Fast.into(), Datum::Categorical(1)),
    ///     ]
    /// );
    ///
    /// let xs = oracle.simulate(
    ///     &vec![Column::Black.into(), Column::Tail.into()],
    ///     &given,
    ///     10,
    ///     None,
    ///     &mut rng,
    /// ).unwrap();
    ///
    /// assert_eq!(xs.len(), 10);
    /// assert!(xs.iter().all(|x| x.len() == 2));
    /// ```
    pub fn simulate(
        &self,
        col_ixs: &[usize],
        given: &Given,
        n: usize,
        states_ixs_opt: Option<Vec<usize>>,
        mut rng: &mut impl Rng,
    ) -> Result<Vec<Vec<Datum>>, error::SimulateError> {
        let ncols = self.ncols();
        if col_ixs.is_empty() {
            return Err(error::SimulateError::NoTargetsError);
        } else if col_ixs.iter().any(|&ix| ix >= ncols) {
            return Err(error::SimulateError::TargetIndexOutOfBoundsError);
        }

        if let Some(ref state_ixs) = states_ixs_opt {
            if state_ixs.iter().any(|&ix| ix > self.nstates()) {
                return Err(error::SimulateError::StateIndexOutOfBoundsError);
            }
        }

        find_given_errors(col_ixs, &self, given).map_err(|err| err.into())?;

        Ok(
            self.simulate_unchecked(
                col_ixs,
                given,
                n,
                states_ixs_opt,
                &mut rng,
            ),
        )
    }

    /// Return the most likely value for a cell in the table along with the
    /// confidence in that imputation.
    ///
    /// Imputation can be done on non-missing cells and will re-predict the
    /// value of the cell rather than returning the existing value. To get
    /// the current value of a cell, use `Oracle::data`.
    ///
    /// # Arguments
    ///
    /// - row_ix: the row index of the cell to impute
    /// - col_ix: the column index of the cell to impute
    /// - with_unc: if `true` compute the uncertainty, otherwise a value of -1
    ///   is returned in the uncertainty spot
    ///
    /// # Returns
    ///
    /// A `(value, uncertainty_option)` tuple. If `with_unc` is `false`,
    /// `uncertainty` is -1.
    ///
    /// # Example
    ///
    /// Impute the value of swims for an dolphin and an polar bear.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::examples::animals::{Column, Row};
    /// use braid::interface::ImputeUncertaintyType;
    /// use braid_stats::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let dolphin_swims = oracle.impute(
    ///     Row::Dolphin.into(),
    ///     Column::Swims.into(),
    ///     Some(ImputeUncertaintyType::JsDivergence)
    /// ).unwrap();
    ///
    /// let bear_swims = oracle.impute(
    ///     Row::PolarBear.into(),
    ///     Column::Swims.into(),
    ///     Some(ImputeUncertaintyType::JsDivergence)
    /// ).unwrap();
    ///
    /// assert_eq!(dolphin_swims.0, Datum::Categorical(1));
    /// assert_eq!(bear_swims.0, Datum::Categorical(1));
    ///
    /// let dolphin_swims_unc = dolphin_swims.1.unwrap();
    /// let bear_swims_unc = bear_swims.1.unwrap();
    ///
    /// // Given that a polar bear is a furry, footed mammal, it's harder to
    /// // model  why we know it swims.
    /// assert!(bear_swims_unc > dolphin_swims_unc);
    /// ```
    pub fn impute(
        &self,
        row_ix: usize,
        col_ix: usize,
        unc_type_opt: Option<ImputeUncertaintyType>,
    ) -> Result<(Datum, Option<f64>), IndexError> {
        if row_ix >= self.nrows() {
            return Err(IndexError::RowIndexOutOfBoundsError);
        } else if col_ix >= self.ncols() {
            return Err(IndexError::ColumnIndexOutOfBoundsError);
        }

        let val: Datum = match self.ftype(col_ix).unwrap() {
            FType::Continuous => {
                let x = utils::continuous_impute(&self.states, row_ix, col_ix);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x = utils::categorical_impute(&self.states, row_ix, col_ix);
                Datum::Categorical(x)
            }
            FType::Labeler => {
                let x = utils::labeler_impute(&self.states, row_ix, col_ix);
                Datum::Label(x)
            }
        };

        let unc_opt = match unc_type_opt {
            Some(unc_type) => {
                Some(self.impute_uncertainty(row_ix, col_ix, unc_type))
            }
            None => None,
        };

        Ok((val, unc_opt))
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
    pub fn predict(
        &self,
        col_ix: usize,
        given: &Given,
        unc_type_opt: Option<PredictUncertaintyType>,
    ) -> Result<(Datum, Option<f64>), error::PredictError> {
        if col_ix >= self.ncols() {
            return Err(error::PredictError::ColumnIndexOutOfBoundsError);
        }

        find_given_errors(&vec![col_ix], &self, &given)
            .map_err(|err| err.into())?;

        let value = match self.ftype(col_ix).unwrap() {
            FType::Continuous => {
                let x = utils::continuous_predict(&self.states, col_ix, &given);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x =
                    utils::categorical_predict(&self.states, col_ix, &given);
                Datum::Categorical(x)
            }
            FType::Labeler => {
                let x = utils::labeler_predict(&self.states, col_ix, &given);
                Datum::Label(x)
            }
        };

        let unc_opt = match unc_type_opt {
            Some(_) => Some(self.predict_uncertainty(col_ix, &given)),
            None => None,
        };

        Ok((value, unc_opt))
    }

    /// Compute the error between the observed data in a feature and the feature
    /// model.
    ///
    /// # Returns
    /// An `(error, centroid)` tuple where error a float in [0, 1], and the
    /// centroid is the centroid of  the error. For continuous features, the
    /// error is derived from the probability integral transform, and for
    /// discrete variables the error is **WRITEME**
    pub fn feature_error(&self, col_ix: usize) -> (f64, f64) {
        // extract the feature from the first state
        let ftr = self.states[0].feature(col_ix);
        let ftype = ftr.ftype();
        // TODO: can this replicated code be macroed away?
        //
        if ftype.is_continuous() {
            let mixtures: Vec<Mixture<Gaussian>> = self
                .states
                .iter()
                .map(|state| state.feature_as_mixture(col_ix).into())
                .collect();
            let mixture = Mixture::combine(mixtures);
            let xs: Vec<f64> = (0..self.nrows())
                .filter_map(|row_ix| self.data.get(row_ix, col_ix).to_f64_opt())
                .collect();
            mixture.sample_error(&xs)
        } else if ftype.is_categorical() {
            let mixtures: Vec<Mixture<Categorical>> = self
                .states
                .iter()
                .map(|state| state.feature_as_mixture(col_ix).into())
                .collect();
            let mixture = Mixture::combine(mixtures);
            let xs: Vec<u8> = (0..self.nrows())
                .filter_map(|row_ix| self.data.get(row_ix, col_ix).to_u8_opt())
                .collect();
            mixture.sample_error(&xs)
        } else {
            panic!("Unsupported feature type");
        }
    }
}

// Private function impls
impl Oracle {
    fn logp_unchecked(
        &self,
        col_ixs: &[usize],
        vals: &Vec<Vec<Datum>>,
        given: &Given,
        states_ixs_opt: Option<Vec<usize>>,
    ) -> Vec<f64> {
        let log_nstates;
        let logps: Vec<Vec<f64>> = match states_ixs_opt {
            Some(state_ixs) => {
                log_nstates = (state_ixs.len() as f64).ln();
                state_ixs
                    .iter()
                    .map(|&ix| {
                        utils::state_logp(
                            &self.states[ix],
                            &col_ixs,
                            &vals,
                            &given,
                        )
                    })
                    .collect()
            }
            None => {
                log_nstates = (self.nstates() as f64).ln();
                self.states
                    .iter()
                    .map(|state| {
                        utils::state_logp(state, &col_ixs, &vals, &given)
                    })
                    .collect()
            }
        };

        transpose(&logps)
            .iter()
            .map(|lps| logsumexp(&lps) - log_nstates)
            .collect()
    }

    fn simulate_unchecked(
        &self,
        col_ixs: &[usize],
        given: &Given,
        n: usize,
        states_ixs_opt: Option<Vec<usize>>,
        mut rng: &mut impl Rng,
    ) -> Vec<Vec<Datum>> {
        let state_ixs: Vec<usize> = match states_ixs_opt {
            Some(state_ixs) => state_ixs,
            None => (0..self.nstates()).collect(),
        };

        let states: Vec<&State> =
            state_ixs.iter().map(|&ix| &self.states[ix]).collect();
        let weights = utils::given_weights(&states, &col_ixs, &given);
        let state_ixer = Categorical::uniform(state_ixs.len());

        (0..n)
            .map(|_| {
                // choose a random state
                let draw_ix: usize = state_ixer.draw(&mut rng);
                let state_ix: usize = state_ixs[draw_ix];
                let state = states[draw_ix];

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

    fn surprisal_unchecked(
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

    // specialization for column pairs. If a specialization is not founds for
    // the specific columns types, will fall back to QMC approximation
    fn dual_entropy(&self, col_a: usize, col_b: usize, n: usize) -> f64 {
        let ftypes = (self.ftype(col_a).unwrap(), self.ftype(col_b).unwrap());
        match ftypes {
            (FType::Categorical, FType::Categorical) => {
                utils::categorical_entropy_dual(col_a, col_b, &self.states)
            }
            (FType::Categorical, FType::Continuous) => {
                utils::categorical_gaussian_entropy_dual(
                    col_a,
                    col_b,
                    &self.states,
                )
            }
            (FType::Continuous, FType::Categorical) => {
                utils::categorical_gaussian_entropy_dual(
                    col_b,
                    col_a,
                    &self.states,
                )
            }
            _ => self.sobol_joint_entropy(&[col_a, col_b], n),
        }
    }

    fn mi_components(
        &self,
        col_a: usize,
        col_b: usize,
        n: usize,
    ) -> MiComponents {
        if col_a == col_b {
            let h_a = utils::entropy_single(col_a, &self.states);
            MiComponents {
                h_a,
                h_b: h_a,
                h_ab: h_a,
            }
        } else {
            let h_a = utils::entropy_single(col_a, &self.states);
            let h_b = utils::entropy_single(col_b, &self.states);
            let h_ab = self.dual_entropy(col_a, col_b, n);
            MiComponents { h_a, h_b, h_ab }
        }
    }

    // Use a Sobol QMC sequence to appropriate joint entropy
    fn sobol_joint_entropy(&self, col_ixs: &[usize], n: usize) -> f64 {
        let (vals, q_recip) =
            utils::gen_sobol_samples(col_ixs, &self.states[0], n);
        let logps = self.logp_unchecked(col_ixs, &vals, &Given::Nothing, None);
        let h: f64 = logps.iter().map(|logp| -logp * logp.exp()).sum();
        h * q_recip / (n as f64)
    }

    fn entropy_unchecked(&self, col_ixs: &[usize], n: usize) -> f64 {
        match col_ixs.len() {
            0 => unreachable!(),
            1 => utils::entropy_single(col_ixs[0], &self.states),
            2 => self.dual_entropy(col_ixs[0], col_ixs[1], n),
            _ => self.sobol_joint_entropy(col_ixs, n),
        }
    }

    // Computes the predictive uncertainty for the datum at (row_ix, col_ix)
    // as mean the pairwise KL divergence between the components to which the
    // datum is assigned.
    //
    // # Notes
    // Impute uncertainty applies only to impute operations where we want to
    // recover a specific missing (or not missing) entry. There is no special
    // handling of non-missing entries.
    //
    // # Arguments
    // - row_ix: the row index
    // - col_ix: the column index
    // - n_samples: the number of samples for the Monte Carlo integral. If
    //   `n_samples` is 0, then pairwise KL divergence will be used, otherwise
    //   JS divergence will be approximated.
    #[inline]
    fn impute_uncertainty(
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

    // Computes the uncertainty associated with predicting the value of a
    // features with optional given conditions. Uses Jensen-Shannon divergence
    // computed on the mixture of mixtures.
    //
    // # Notes
    // Predict uncertainty applies only to prediction of hypothetical values,
    // and not to imputation of in-table values.
    //
    // # Arguments
    // - col_ix: the column index
    // - given_opt: an optional list of (column index, value) tuples
    //   designating other observations on which to condition the prediciton
    #[inline]
    pub fn predict_uncertainty(&self, col_ix: usize, given: &Given) -> f64 {
        utils::predict_uncertainty(&self.states, col_ix, &given)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    use braid_stats::MixtureType;
    use rv::dist::Mixture;
    use std::path::Path;

    fn oracle_from_yaml<P: AsRef<Path>>(filenames: Vec<P>) -> Oracle {
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
            "resources/test/small/small-state-1.yaml",
            "resources/test/small/small-state-2.yaml",
            "resources/test/small/small-state-3.yaml",
        ];

        oracle_from_yaml(filenames)
    }

    fn get_entropy_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/entropy/entropy-state-1.yaml",
            "resources/test/entropy/entropy-state-2.yaml",
        ];
        oracle_from_yaml(filenames)
    }

    #[test]
    fn single_continuous_column_logp() {
        let oracle = get_single_continuous_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp =
            oracle.logp(&vec![0], &vals, &Given::Nothing, None).unwrap()[0];

        assert_relative_eq!(logp, -2.7941051646651953, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_logp_state_0() {
        let oracle = get_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp = oracle
            .logp(&vec![0], &vals, &Given::Nothing, Some(vec![0]))
            .unwrap()[0];

        assert_relative_eq!(logp, -1.223532985437053, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_logp_duplicated_states() {
        let oracle = get_duplicate_single_continuous_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp =
            oracle.logp(&vec![0], &vals, &Given::Nothing, None).unwrap()[0];

        assert_relative_eq!(logp, -2.7941051646651953, epsilon = TOL);
    }

    #[test]
    #[ignore]
    fn mutual_information_smoke() {
        let oracle = get_oracle_from_yaml();

        let mi_01 = oracle.mi(0, 1, 10_000, MiType::Normed).unwrap();
        let mi_02 = oracle.mi(0, 2, 10_000, MiType::Normed).unwrap();
        let mi_12 = oracle.mi(1, 2, 10_000, MiType::Normed).unwrap();

        println!("01 {}", mi_01);
        println!("02 {}", mi_02);
        println!("12 {}", mi_12);
        assert!(mi_01 > 0.0);
        assert!(mi_02 > 0.0);
        assert!(mi_12 > 0.0);
    }

    #[test]
    fn surpisal_value_1() {
        let oracle = get_oracle_from_yaml();
        let s = oracle
            .surprisal(&Datum::Continuous(1.2), 3, 1)
            .unwrap()
            .unwrap();
        assert_relative_eq!(s, 1.7739195803316758, epsilon = 10E-7);
    }

    #[test]
    fn surpisal_value_2() {
        let oracle = get_oracle_from_yaml();
        let s = oracle
            .surprisal(&Datum::Continuous(0.1), 1, 0)
            .unwrap()
            .unwrap();
        assert_relative_eq!(s, 0.62084325305231269, epsilon = 10E-7);
    }

    #[test]
    fn kl_impute_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let u =
            oracle.impute_uncertainty(0, 1, ImputeUncertaintyType::PairwiseKl);
        assert!(u > 0.0);
    }

    #[test]
    fn js_impute_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
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
        let u = oracle.predict_uncertainty(0, &Given::Nothing);
        assert!(u > 0.0);
    }

    #[test]
    fn predict_uncertainty_smoke_with_given() {
        let oracle = get_oracle_from_yaml();
        let given = Given::Conditions(vec![(1, Datum::Continuous(2.5))]);
        let u = oracle.predict_uncertainty(0, &given);
        assert!(u > 0.0);
    }

    // FIXME: Make this test run w/ only a csv. It hard to maintain a test that
    // requires a re-analysis to generate the assets. Ignoring for now.
    // NOTE: though the data go to about 4, the max uncertainty for these data
    // seems to hit about at 3.0 when the two branches are completely
    // separated, which makes sense.
    #[ignore]
    #[test]
    fn predict_uncertainty_calipers() {
        use std::f64::NEG_INFINITY;
        let oracle =
            Oracle::load(&Path::new("resources/test/calipers.braid")).unwrap();
        let xs = vec![1.0, 2.0, 2.5, 3.0];
        let (_, uncertainty_increasing) =
            xs.iter().fold((NEG_INFINITY, true), |acc, x| {
                let given = Given::Conditions(vec![(0, Datum::Continuous(*x))]);
                let unc = oracle.predict_uncertainty(1, &given);
                println!("Unc y|x={} is {}", x, unc);
                if unc > acc.0 && acc.1 {
                    (unc, true)
                } else {
                    (unc, false)
                }
            });
        assert!(uncertainty_increasing);
    }

    #[test]
    fn mixture_and_oracle_logp_equivalence_categorical() {
        let oracle = get_entropy_oracle_from_yaml();

        let mm: Mixture<Categorical> = {
            let mixtures: Vec<_> = oracle
                .states
                .iter()
                .map(|s| s.feature_as_mixture(2))
                .collect();
            match MixtureType::combine(mixtures) {
                MixtureType::Categorical(mm) => mm,
                _ => panic!("bad mixture type"),
            }
        };

        for x in 0..4 {
            let y = Datum::Categorical(x as u8);
            let logp_mm = mm.ln_f(&(x as usize));
            let logp_or = oracle
                .logp(&vec![2], &vec![vec![y]], &Given::Nothing, None)
                .unwrap()[0];
            assert_relative_eq!(logp_or, logp_mm, epsilon = 1E-12);
        }
    }

    #[test]
    fn mixture_and_oracle_logp_equivalence_gaussian() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let mm: Mixture<Gaussian> = {
            let mixtures: Vec<_> = oracle
                .states
                .iter()
                .map(|s| s.feature_as_mixture(1))
                .collect();
            match MixtureType::combine(mixtures) {
                MixtureType::Gaussian(mm) => mm,
                _ => panic!("bad mixture type"),
            }
        };

        for _ in 0..1000 {
            let x: f64 = {
                let u: f64 = rng.gen();
                u * 3.0
            };
            let y = Datum::Continuous(x);
            let logp_mm = mm.ln_f(&x);
            let logp_or = oracle
                .logp(&vec![1], &vec![vec![y]], &Given::Nothing, None)
                .unwrap()[0];
            assert_relative_eq!(logp_or, logp_mm, epsilon = 1E-12);
        }
    }

    #[test]
    fn recreate_doctest_mi_failure() {
        use crate::examples::animals::Column;
        use crate::examples::Example;
        use crate::interface::MiType;

        let oracle = Example::Animals.oracle().unwrap();

        let mi_flippers = oracle
            .mi(
                Column::Swims.into(),
                Column::Flippers.into(),
                1000,
                MiType::Iqr,
            )
            .unwrap();

        let mi_fast = oracle
            .mi(Column::Swims.into(), Column::Fast.into(), 1000, MiType::Iqr)
            .unwrap();

        assert!(mi_flippers > mi_fast);
    }

    #[test]
    fn mixture_and_oracle_logp_equivalence_animals_single_state() {
        use crate::examples::Example;

        let oracle = Example::Animals.oracle().unwrap();

        for (ix, state) in oracle.states.iter().enumerate() {
            for col_ix in 0..oracle.ncols() {
                let mm = match state.feature_as_mixture(col_ix) {
                    MixtureType::Categorical(mm) => mm,
                    _ => panic!("Invalid MixtureType"),
                };
                for val in 0..2 {
                    let logp_mm = mm.ln_f(&(val as usize));
                    let datum = Datum::Categorical(val as u8);
                    let logp_or = oracle
                        .logp(
                            &vec![col_ix],
                            &vec![vec![datum]],
                            &Given::Nothing,
                            Some(vec![ix]),
                        )
                        .unwrap()[0];
                    assert_relative_eq!(logp_or, logp_mm, epsilon = 1E-12);
                }
            }
        }
    }

    #[test]
    fn pw_and_conditional_entropy_equivalence_animals() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        let ncols = oracle.ncols();
        let mut col_pairs: Vec<(usize, usize)> = Vec::new();
        let mut entropies: Vec<f64> = Vec::new();
        for col_a in 0..ncols {
            for col_b in 0..ncols {
                if col_a != col_b {
                    col_pairs.push((col_a, col_b));
                    let ce = oracle
                        .conditional_entropy(col_a, &vec![col_b], 1000)
                        .unwrap();
                    entropies.push(ce);
                }
            }
        }

        let entropies_pw = oracle
            .conditional_entropy_pw(
                &col_pairs,
                1000,
                ConditionalEntropyType::UnNormed,
            )
            .unwrap();

        entropies
            .iter()
            .zip(entropies_pw.iter())
            .for_each(|(h, h_pw)| {
                assert_relative_eq!(h, h_pw, epsilon = 1E-12);
            })
    }

    #[test]
    fn pw_and_info_prop_equivalence_animals() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        let ncols = oracle.ncols();
        let mut col_pairs: Vec<(usize, usize)> = Vec::new();
        let mut entropies: Vec<f64> = Vec::new();
        for col_a in 0..ncols {
            for col_b in 0..ncols {
                if col_a != col_b {
                    col_pairs.push((col_a, col_b));
                    let ce = oracle
                        .info_prop(&vec![col_a], &vec![col_b], 1000)
                        .unwrap();
                    entropies.push(ce);
                }
            }
        }

        let entropies_pw = oracle
            .conditional_entropy_pw(
                &col_pairs,
                1000,
                ConditionalEntropyType::InfoProp,
            )
            .unwrap();

        entropies
            .iter()
            .zip(entropies_pw.iter())
            .for_each(|(h, h_pw)| {
                assert_relative_eq!(h, h_pw, epsilon = 1E-12);
            })
    }
}

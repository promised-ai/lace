use std::collections::BTreeSet;

use braid_cc::feature::{FType, Feature};
use braid_cc::state::{State, StateDiagnostics};
use braid_codebook::Codebook;
use braid_data::{Datum, SummaryStatistics};
use braid_stats::SampleError;
use braid_utils::logsumexp;
use rand::Rng;
use rayon::prelude::*;
use rv::dist::{Categorical, Gaussian, Mixture};
use rv::traits::Rv;

use super::error::{self, IndexError};
use super::utils;
use super::utils::ColumnMaximumLogpCache;
use super::validation::{find_given_errors, find_value_conflicts};
use crate::interface::oracle::error::SurprisalError;
use crate::interface::oracle::{
    ConditionalEntropyType, ImputeUncertaintyType, MiComponents, MiType,
    PredictUncertaintyType,
};
use crate::interface::{CanOracle, Given};
use crate::{ColumnIndex, RowIndex};

macro_rules! col_indices_ok  {
    ($n_cols:expr, $col_ixs:expr, $($err_variant:tt)+) => {{
       $col_ixs.iter().try_for_each(|&col_ix| {
           if col_ix >= $n_cols {
               Err($($err_variant)+ { col_ix, n_cols: $n_cols }) } else {
               Ok(())
           }
       })
    }}
}

macro_rules! state_indices_ok  {
    ($n_states:expr, $state_ixs:expr, $($err_variant:tt)+) => {{
       $state_ixs.iter().try_for_each(|&state_ix| {
           if state_ix >= $n_states {
               Err($($err_variant)+ { state_ix, n_states: $n_states })
           } else {
               Ok(())
           }
       })
    }}
}

macro_rules! feature_err_arm {
    ($this: ident, $col_ix: ident,$mix_type: ty, $data_type: ty, $converter: expr) => {{
        let mixtures: Vec<Mixture<$mix_type>> = $this
            .states()
            .iter()
            .map(|state| state.feature_as_mixture($col_ix).into())
            .collect();
        let mixture = Mixture::combine(mixtures);
        let xs: Vec<$data_type> = (0..$this.n_rows())
            .filter_map(|row_ix| $converter(row_ix, $col_ix))
            .collect();
        mixture.sample_error(&xs)
    }};
}

pub trait OracleT: CanOracle {
    /// Returns the diagnostics for each state
    fn state_diagnostics(&self) -> Vec<StateDiagnostics>;

    /// Returns a tuple containing the number of rows, the number of columns,
    /// and the number of states
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let shape = oracle.shape();
    ///
    /// assert_eq!(shape, (50, 85, 8));
    /// ```
    fn shape(&self) -> (usize, usize, usize) {
        (self.n_rows(), self.n_cols(), self.n_states())
    }

    /// Returns true if the object is empty, having no structure to analyze.
    fn is_empty(&self) -> bool {
        self.states()[0].is_empty()
    }

    /// Return the FType of the column `col_ix`
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_cc::feature::FType;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let ftype = oracle.ftype("swims").unwrap();
    ///
    /// assert_eq!(ftype, FType::Categorical);
    /// ```
    fn ftype<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
    ) -> Result<FType, IndexError>;

    /// Returns a vector of the feature types of each row
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let ftypes = oracle.ftypes();
    ///
    /// assert!(ftypes.iter().all(|ftype| ftype.is_categorical()));
    /// ```
    fn ftypes(&self) -> Vec<FType> {
        (0..self.n_cols())
            .map(|col_ix| self.ftype(col_ix).unwrap())
            .collect()
    }

    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    /// use braid_data::SummaryStatistics;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let swims_summary = oracle.summarize_col("swims").unwrap();
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
    fn summarize_col<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
    ) -> Result<SummaryStatistics, IndexError> {
        extract_colix(col_ix, self.codebook())
            .map(|col_ix| self.summarize_feature(col_ix))
    }

    /// Estimated dependence probability between `col_a` and `col_b`
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let depprob_flippers = oracle.depprob(
    ///     "swims",
    ///     "flippers",
    /// ).unwrap();
    ///
    /// let depprob_fast = oracle.depprob(
    ///     "swims",
    ///     "fast",
    /// ).unwrap();
    ///
    /// assert!(depprob_flippers > depprob_fast);
    /// ```
    fn depprob<Ix: Into<ColumnIndex>>(
        &self,
        col_a: Ix,
        col_b: Ix,
    ) -> Result<f64, IndexError>;

    /// Compute dependence probability for a list of column pairs.
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let depprobs = oracle.depprob_pw(&vec![(1, 12), (3, 2)]).unwrap();
    ///
    /// assert_eq!(depprobs.len(), 2);
    /// assert_eq!(depprobs[0], oracle.depprob(1, 12).unwrap());
    /// assert_eq!(depprobs[1], oracle.depprob(3, 2).unwrap());
    /// ```
    fn depprob_pw<'x, Ix>(
        &self,
        pairs: &'x [(Ix, Ix)],
    ) -> Result<Vec<f64>, IndexError>
    where
        Ix: Into<ColumnIndex> + Clone,
        &'x [(Ix, Ix)]: IntoParallelIterator<Item = &'x (Ix, Ix)>,
    {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        pairs
            .par_iter()
            .map(|pair| {
                extract_col_pair(pair, self.codebook())
                    .and_then(|(col_a, col_b)| self.depprob(col_a, col_b))
            })
            .collect()
    }

    fn _rowsim_validation(
        &self,
        row_a: usize,
        row_b: usize,
        wrt: &Option<&Vec<usize>>,
    ) -> Result<(), error::RowSimError> {
        let n_rows = self.n_rows();
        if row_a >= n_rows {
            return Err(error::RowSimError::Index(
                IndexError::RowIndexOutOfBounds {
                    row_ix: row_a,
                    n_rows,
                },
            ));
        } else if row_b >= n_rows {
            return Err(error::RowSimError::Index(
                IndexError::RowIndexOutOfBounds {
                    row_ix: row_b,
                    n_rows,
                },
            ));
        }

        if let Some(col_ixs) = wrt {
            if col_ixs.is_empty() {
                return Err(error::RowSimError::EmptyWrt);
            }
        }

        Ok(())
    }

    /// Estimated row similarity between `row_a` and `row_b`
    ///
    /// # Arguments
    /// - row_a: the first row index
    /// - row_b: the second row index
    /// - wrt: an optional vector of column indices to constrain the similarity.
    ///   Only the view to which the columns in `wrt` are assigned will be
    ///   considered in the similarity calculation
    /// - col_weighted: if `true` similarity will be weighted by the number of
    ///   columns rather than the number of views. In this mode rows with more
    ///   cells in the same categories will have higher weight.
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let wrt: Option<&[usize]> = None;
    /// let rowsim = oracle.rowsim(
    ///     "wolf",
    ///     "collie",
    ///     wrt,
    ///     false,
    /// ).unwrap();
    ///
    /// assert!(rowsim >= 0.0 && rowsim <= 1.0);
    /// ```
    /// Adding context with `wrt` (with respect to):
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::OracleT;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// # let wrt: Option<&[usize]> = None;
    /// # let rowsim = oracle.rowsim(
    /// #     "wolf",
    /// #     "collie",
    /// #     wrt,
    /// #     false,
    /// # ).unwrap();
    ///
    /// let rowsim_wrt = oracle.rowsim(
    ///     "wolf",
    ///     "collie",
    ///     Some(&["swims"]),
    ///     false,
    /// ).unwrap();
    ///
    /// assert_ne!(rowsim, rowsim_wrt);
    /// ```
    fn rowsim<RIx: Into<RowIndex>, CIx: Into<ColumnIndex> + Clone>(
        &self,
        row_a: RIx,
        row_b: RIx,
        wrt: Option<&[CIx]>,
        col_weighted: bool,
    ) -> Result<f64, error::RowSimError>;

    /// Compute row similarity for pairs of rows
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let wrt: Option<&[usize]> = None;
    /// let rowsims = oracle.rowsim_pw(
    ///     &[
    ///         ("gorilla", "spider+monkey"),
    ///         ("gorilla", "skunk"),
    ///     ],
    ///     wrt,
    ///     false,
    /// ).unwrap();
    ///
    /// assert!(rowsims.iter().all(|&rowsim| 0.0 <= rowsim && rowsim <= 1.0));
    /// ```
    fn rowsim_pw<'x, RIx, CIx>(
        &self,
        pairs: &'x [(RIx, RIx)],
        wrt: Option<&[CIx]>,
        col_weighted: bool,
    ) -> Result<Vec<f64>, error::RowSimError>
    where
        RIx: Into<RowIndex> + Clone,
        CIx: Into<ColumnIndex> + Clone + Sync,
        &'x [(RIx, RIx)]: IntoParallelIterator<Item = &'x (RIx, RIx)>,
    {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        // TODO: Speed up by recomputing the view indices for each state
        pairs
            .par_iter()
            .map(|pair| {
                extract_row_pair(pair, self.codebook())
                    .map_err(error::RowSimError::Index)
                    .and_then(|(row_a, row_b)| {
                        self.rowsim(row_a, row_b, wrt, col_weighted)
                    })
            })
            .collect()
    }

    /// Determine the relative novelty of a row.
    ///
    /// Novelty is defined as the reciprocal of the mean size of categories (as
    /// a proportion of the total number of data) to which the row belongs. If
    /// a row is in smaller categories, it will have a higher novelty.
    ///
    /// # Notes
    /// Novelty is contextual; it must be compared to the novelty of all other
    /// rows. The mean novelty score will increase as the data become more
    /// divided. For example, if there is one view with two even categories,
    /// each row's novelty will be 0.5; if there are four even categories, the
    /// mean novelty score will be 0.75.
    ///
    /// # Example
    /// Dolphins are more novel than rats
    ///
    /// ```no_run
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let wrt: Option<&[usize]> = None;
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let novelty_dolphin = oracle.novelty("dolphin", wrt).unwrap();
    /// let novelty_rat = oracle.novelty("rat", wrt).unwrap();
    ///
    /// assert!(novelty_rat < novelty_dolphin);
    /// ```
    ///
    /// Dolphins are more novel than rats with respect to their swimming.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::OracleT;
    /// # let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let wrt = vec!["swims"];
    ///
    /// let novelty_rat = oracle.novelty("rat", Some(&wrt)).unwrap();
    /// let novelty_dolphin = oracle.novelty("dolphin", Some(&wrt)).unwrap();
    ///
    /// assert!(novelty_dolphin > novelty_rat);
    /// ```
    fn novelty<RIx: Into<RowIndex>, CIx: Into<ColumnIndex> + Clone>(
        &self,
        row_ix: RIx,
        wrt: Option<&[CIx]>,
    ) -> Result<f64, IndexError>;

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
    /// use braid::OracleT;
    /// use braid::MiType;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mi_flippers = oracle.mi(
    ///     "swims",
    ///     "flippers",
    ///     1000,
    ///     MiType::Iqr,
    /// ).unwrap();
    ///
    /// let mi_fast = oracle.mi(
    ///     "swims",
    ///     "fast",
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
    /// # use braid::OracleT;
    /// # use braid::MiType;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// let mi_self = oracle.mi(
    ///     "swims",
    ///     "swims",
    ///     1000,
    ///     MiType::Iqr,
    /// ).unwrap();
    ///
    /// assert_eq!(mi_self, 1.0);
    /// ```
    fn mi<Ix: Into<ColumnIndex>>(
        &self,
        col_a: Ix,
        col_b: Ix,
        n: usize,
        mi_type: MiType,
    ) -> Result<f64, error::MiError> {
        if n == 0 {
            return Err(error::MiError::NIsZero);
        }

        let col_a = extract_colix(col_a, self.codebook())?;
        let col_b = extract_colix(col_b, self.codebook())?;

        let mi_cpnts = self._mi_components(col_a, col_b, n);
        Ok(mi_cpnts.compute(mi_type))
    }

    /// Compute mutual information over pairs of columns
    ///
    /// # Notes
    ///
    /// This function has special optimizations over computing oracle::mi for
    /// pairs manually.
    fn mi_pw<Ix: Into<ColumnIndex> + Clone>(
        &self,
        pairs: &[(Ix, Ix)],
        n: usize,
        mi_type: MiType,
    ) -> Result<Vec<f64>, error::MiError>;

    /// Estimate joint entropy
    ///
    /// # Notes
    /// The computation is exact under certain circumstances, otherwise the
    /// quantity is approximated via Monte Carlo integration.
    ///
    /// - All columns are categorical, in which case the exact answer is
    ///   computed via enumeration. The user should be aware combinatorial
    ///   expansion of the terms in the summation.
    /// - There is only one index in col_ixs and that column is categorical,
    ///   gaussian, or labeler.
    /// - There are two columns and one is categorical and the other is
    ///   gaussian
    ///
    /// # Arguments
    /// - col_ixs: vector of column indices
    /// - n: number of samples for the Monte Carlo integral.
    ///
    /// # Examples
    ///
    /// There is more information in the swims column than in the blue column
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    /// use braid::MiType;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// // Close to uniformly distributed -> high entropy
    /// let h_swims = oracle.entropy(
    ///     &["swims"],
    ///     10_000,
    /// ).unwrap();
    ///
    /// // Close to deterministic -> low entropy
    /// let h_blue = oracle.entropy(
    ///     &["blue"],
    ///     10_000,
    /// ).unwrap();
    ///
    /// assert!(h_blue < h_swims);
    /// ```
    ///
    /// The `n` argument isn't used for a single categorical column because
    /// the exact computation is used.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid::OracleT;
    /// # use braid::MiType;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// let h_swims_10k = oracle.entropy(
    ///     &["swims"],
    ///     10_000,
    /// ).unwrap();
    ///
    /// let h_swims_0 = oracle.entropy(
    ///     &["swims"],
    ///     1,
    /// ).unwrap();
    ///
    /// assert!((h_swims_10k - h_swims_0).abs() < 1E-12);
    /// ```
    fn entropy<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        n: usize,
    ) -> Result<f64, error::EntropyError> {
        if col_ixs.is_empty() {
            return Err(error::EntropyError::NoTargetColumns);
        } else if n == 0 {
            return Err(error::EntropyError::NIsZero);
        }

        let col_ixs = extract_colixs(col_ixs, self.codebook())?;

        Ok(self._entropy_unchecked(&col_ixs, n))
    }

    /// Determine the set of predictors that most efficiently account for the
    /// most information in a set of target columns.
    ///
    /// # Notes
    /// The estimates will be bad if the number of samples is too low to fill
    /// the space. This will be particularly apparent in large numbers of
    /// categorical variables where not filling the space means missing out on
    /// entire classes. If you notice large jumps in the running info_prop (it
    /// should be roughly `log(n)`), then you are having bad error and will
    /// need to up the number of samples.  **The max recommended number of
    /// predictors plus targets is 10**.
    ///
    /// # Arguments
    /// - cols_t: The target column indices. The ones you want to predict.
    /// - max_predictors: The max number of predictors to search.
    /// - n_qmc_samples: The number of QMC samples to use for entropy
    ///   estimation
    ///
    /// # Returns
    /// A Vec of (col_ix, info_prop). The first column index is the column that is the single best
    /// predictor of the targets. The additional columns in the sequence are the columns added to
    /// the predictor set that maximizes the prediction. The information proportions are the
    /// proportions of information accounted for by the predictors with that column added to the
    /// set.
    ///
    /// # Example
    ///
    /// Which four columns should I choose to best predict whether an animals swims
    ///
    /// ```
    /// # use braid::examples::Example;
    /// # use braid_cc::feature::FType;
    /// use braid::examples::animals::Column;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let predictors = oracle.predictor_search(
    ///     &vec!["swims"],
    ///     4,
    ///     10_000
    /// ).unwrap();
    ///
    /// // We asked for four predictors, so we get four.
    /// assert_eq!(predictors.len(), 4);
    ///
    /// // Whether something lives in water is the single best predictor of
    /// // whether something swims.
    /// let water: usize = Column::Water.into();
    /// assert_eq!(predictors[0].0, water);
    ///
    /// // All information proportions, without runaway approximation error,
    /// // should be in [0, 1].
    /// for (_col_ix, info_prop) in &predictors {
    ///     assert!(0.0 < *info_prop && *info_prop < 1.0)
    /// }
    ///
    /// // As we add predictors, the information proportions increase
    /// // monotonically
    /// for i in 1..4 {
    ///     assert!(predictors[i-1].1 < predictors[i].1);
    /// }
    /// ```
    fn predictor_search<Ix: Into<ColumnIndex> + Clone>(
        &self,
        cols_t: &[Ix],
        max_predictors: usize,
        n_qmc_samples: usize,
    ) -> Result<Vec<(usize, f64)>, IndexError> {
        let cols_t = extract_colixs(cols_t, self.codebook())?;
        // TODO: Faster algorithm with less sampler error
        // Perhaps using an algorithm looking only at the mutual information
        // between the candidate and the targets, and the candidate and the last
        // best column?
        let mut to_search: BTreeSet<usize> = {
            let targets: BTreeSet<usize> = cols_t.iter().cloned().collect();
            (0..self.n_cols())
                .filter(|ix| !targets.contains(ix))
                .collect()
        };

        let n_predictors = max_predictors.min(to_search.len());

        let mut predictors: Vec<usize> = Vec::new();

        Ok((0..n_predictors)
            .map(|_| {
                let best_col: (usize, f64) = to_search
                    .par_iter()
                    .map(|&ix| {
                        let mut p_local = predictors.clone();
                        p_local.push(ix);
                        let info_prop = self
                            .info_prop(&cols_t, &p_local, n_qmc_samples)
                            .unwrap();
                        (ix, info_prop)
                    })
                    .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                    .unwrap();

                assert!(
                    to_search.remove(&best_col.0),
                    "The best column was not in the search"
                );

                predictors.push(best_col.0);
                best_col
            })
            .collect())
    }

    /// Compute the proportion of information in `cols_t` accounted for by
    /// `cols_x`.
    ///
    /// # Arguments
    /// - cols_t: The target columns. Typically the target of a prediction.
    /// - cols_x: The predictor columns.
    /// - n: the number of samples for the Monte Carlo integral. Make n high
    ///   enough to integrate a function with as many dimensions as there
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
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let ip_flippers = oracle.info_prop(
    ///     &["swims"],
    ///     &["flippers"],
    ///     1000,
    /// ).unwrap();
    ///
    /// let ip_fast = oracle.info_prop(
    ///     &["swims"],
    ///     &["fast"],
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
    /// # use braid::OracleT;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// # let ip_flippers = oracle.info_prop(
    /// #     &["swims"],
    /// #     &["flippers"],
    /// #     1000,
    /// # ).unwrap();
    /// let ip_flippers_coastal = oracle.info_prop(
    ///     &["swims"],
    ///     &["flippers", "coastal"],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(ip_flippers < ip_flippers_coastal);
    /// assert!(ip_flippers_coastal <= 1.0);
    ///
    /// let ip_flippers_coastal_fast = oracle.info_prop(
    ///     &["swims"],
    ///     &["flippers", "coastal", "fast"],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(ip_flippers_coastal < ip_flippers_coastal_fast);
    /// assert!(ip_flippers_coastal_fast <= 1.0);
    /// ```
    fn info_prop<IxT, IxX>(
        &self,
        cols_t: &[IxT],
        cols_x: &[IxX],
        n: usize,
    ) -> Result<f64, error::InfoPropError>
    where
        IxT: Into<ColumnIndex> + Clone,
        IxX: Into<ColumnIndex> + Clone,
    {
        if n == 0 {
            return Err(error::InfoPropError::NIsZero);
        } else if cols_t.is_empty() {
            return Err(error::InfoPropError::NoTargetColumns);
        } else if cols_x.is_empty() {
            return Err(error::InfoPropError::NoPredictorColumns);
        }

        let cols_t = extract_colixs(cols_t, self.codebook())
            .map_err(error::InfoPropError::TargetIndexOutOfBounds)?;
        let cols_x = extract_colixs(cols_x, self.codebook())
            .map_err(error::InfoPropError::PredictorIndexOutOfBounds)?;

        let all_cols: Vec<usize> = {
            let mut cols = cols_t.clone();
            cols.extend_from_slice(&cols_x);
            cols
        };

        // The target column is among the predictors, which means that all the
        // information is recovered.
        if all_cols.len() != cols_x.len() + cols_t.len() {
            Ok(1.0)
        } else {
            let h_all = self._entropy_unchecked(&all_cols, n);
            let h_t = self._entropy_unchecked(&cols_t, n);
            let h_x = self._entropy_unchecked(&cols_x, n);

            Ok((h_t + h_x - h_all) / h_t)
        }
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
    /// use braid::OracleT;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mi_flippers = oracle.conditional_entropy(
    ///     "swims",
    ///     &["flippers"],
    ///     1000,
    /// ).unwrap();
    ///
    /// let mi_fast_tail = oracle.conditional_entropy(
    ///     "swims",
    ///     &["fast", "tail"],
    ///     1000,
    /// ).unwrap();
    ///
    /// assert!(mi_flippers < mi_fast_tail);
    /// ```
    fn conditional_entropy<IxT, IxX>(
        &self,
        col_t: IxT,
        cols_x: &[IxX],
        n: usize,
    ) -> Result<f64, error::ConditionalEntropyError>
    where
        IxT: Into<ColumnIndex>,
        IxX: Into<ColumnIndex> + Clone,
    {
        let col_t = extract_colix(col_t, self.codebook())
            .map_err(error::ConditionalEntropyError::TargetIndexOutOfBounds)?;

        let cols_x = extract_colixs(cols_x, self.codebook()).map_err(
            error::ConditionalEntropyError::PredictorIndexOutOfBounds,
        )?;

        if n == 0 {
            return Err(error::ConditionalEntropyError::NIsZero);
        };

        if cols_x.is_empty() {
            return Err(error::ConditionalEntropyError::NoPredictorColumns);
        }

        // The target is a predictor, which means there is no left over entropy
        if cols_x.iter().any(|&ix| ix == col_t) {
            return Ok(0.0);
        }

        let all_cols: Vec<_> = {
            let mut col_ixs: BTreeSet<usize> = BTreeSet::new();
            col_ixs.insert(col_t);
            cols_x.iter().try_for_each(|&col_ix| {
                // insert returns true if col_ix is new
                if col_ixs.insert(col_ix) {
                    Ok(())
                } else {
                    Err(error::ConditionalEntropyError::DuplicatePredictors { col_ix })
                }
            })
            .map(|_| col_ixs.iter().cloned().collect())
        }?;

        let h_x = self._entropy_unchecked(&cols_x, n);
        let h_all = self._entropy_unchecked(&all_cols, n);

        Ok(h_all - h_x)
    }

    /// Pairwise copmutation of conditional entreopy or information proportion
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    /// use braid::ConditionalEntropyType;
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
    /// # use braid::OracleT;
    /// # use braid::examples::animals::Column;
    /// # use braid::ConditionalEntropyType;
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
    fn conditional_entropy_pw<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_pairs: &[(Ix, Ix)],
        n: usize,
        kind: ConditionalEntropyType,
    ) -> Result<Vec<f64>, error::ConditionalEntropyError>;

    /// Negative log PDF/PMF of a datum, x, in a specific cell of the table at
    /// position row_ix, col_ix.
    ///
    /// `surprisal` is different from `logp` in that it works only on cells
    /// that exist in the table. `logp` works on hypothetical data that have
    /// not been inserted into the table. Because the data in a cell is modeled
    /// as a result of the running of the inference algorithm, the likelihood of
    /// any cell is implicitly conditioned on all other cells in the table,
    /// therefore `surprisal` does not accept conditions.
    ///
    /// # Notes
    /// To compute surprisal of non-inserted data, use `-logp(..)`.
    ///
    /// # Arguments
    /// - x: the value of which to compute the surprisal
    /// - row_ix: The row index of `x`
    /// - col_ix: column index of `x`
    /// - state_ixs: The optional state indices over which to compute
    ///   surprisal. If `None`, use all states.
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
    /// use braid::OracleT;
    /// use braid_data::Datum;
    /// use braid::examples::animals::{Column, Row};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let present = Datum::Categorical(1);
    ///
    /// let s_pig = oracle.surprisal(
    ///     &present,
    ///     "pig",
    ///     "fierce",
    ///     None,
    /// ).unwrap();
    ///
    /// let s_lion = oracle.surprisal(
    ///     &present,
    ///     "lion",
    ///     "fierce",
    ///     None,
    /// ).unwrap();
    ///
    /// assert!(s_pig > s_lion);
    /// ```
    fn surprisal<RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        x: &Datum,
        row_ix: RIx,
        col_ix: CIx,
        state_ixs: Option<Vec<usize>>,
    ) -> Result<Option<f64>, error::SurprisalError> {
        let row_ix = extract_rowix(row_ix, self.codebook())?;
        let col_ix = extract_colix(col_ix, self.codebook())?;

        let ftype_compat =
            self.ftype(col_ix).map(|ftype| ftype.datum_compatible(x))?;

        if let Some(ref ixs) = state_ixs {
            state_indices_ok!(
                self.n_states(),
                ixs,
                error::SurprisalError::StateIndexOutOfBounds
            )?;
        }

        if !ftype_compat.0 {
            return Err(error::SurprisalError::InvalidDatumForColumn {
                col_ix,
                ftype_req: ftype_compat.1.ftype_req,
                ftype: ftype_compat.1.ftype,
            });
        }

        Ok(self._surprisal_unchecked(x, row_ix, col_ix, state_ixs))
    }

    /// Get the surprisal of the datum in a cell.
    ///
    /// # Arguments
    /// - row_ix: The hypothetical row index of the cell.
    /// - col_ix: The hypothetical column index of the cell.
    /// - state_ixs: The optional state indices over which to compute
    ///   surprisal. If `None`, use all states.
    ///
    /// # Example
    ///
    /// A pig is fierce, which is more surprising than a lion being fierce.
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let s_pig = oracle.self_surprisal(
    ///     "pig",
    ///     "fierce",
    ///     None,
    /// ).unwrap();
    ///
    /// let s_lion = oracle.self_surprisal(
    ///     "lion",
    ///     "fierce",
    ///     None,
    /// ).unwrap();
    ///
    /// assert!(s_pig > s_lion);
    /// ```
    fn self_surprisal<RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        state_ixs: Option<Vec<usize>>,
    ) -> Result<Option<f64>, error::SurprisalError> {
        let row_ix = extract_rowix(row_ix, self.codebook())?;
        let col_ix = extract_colix(col_ix, self.codebook())?;

        self.datum(row_ix, col_ix)
            .map_err(SurprisalError::from)
            .map(|x| self._surprisal_unchecked(&x, row_ix, col_ix, state_ixs))
    }

    /// Get the datum at an index
    ///
    /// # Example
    ///
    /// ```
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    /// use braid_data::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let x = oracle.datum("pig", "fierce").unwrap();
    ///
    /// assert_eq!(x, Datum::Categorical(1));
    /// ```
    fn datum<RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
    ) -> Result<Datum, IndexError> {
        let row_ix = extract_rowix(row_ix, self.codebook())?;
        let col_ix = extract_colix(col_ix, self.codebook())?;
        Ok(self.cell(row_ix, col_ix))
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
    /// use braid::OracleT;
    /// use braid_data::Datum;
    /// use braid::Given;
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let logp_swims = oracle.logp(
    ///     &["swims"],
    ///     &[vec![Datum::Categorical(0)], vec![Datum::Categorical(1)]],
    ///     &Given::Nothing,
    ///     None,
    /// ).unwrap();
    ///
    /// let logp_swims_given_flippers = oracle.logp(
    ///     &["swims"],
    ///     &[vec![Datum::Categorical(0)], vec![Datum::Categorical(1)]],
    ///     &Given::Conditions(
    ///         vec![(Column::Flippers.into(), Datum::Categorical(1))]
    ///     ),
    ///     None,
    /// ).unwrap();
    ///
    /// // Also: exhaustive probabilities should sum to one.
    /// assert!(logp_swims[1] < logp_swims_given_flippers[1]);
    ///
    /// let sum_p = logp_swims
    ///     .iter()
    ///     .map(|lp| lp.exp())
    ///     .sum::<f64>();
    ///
    /// assert!((sum_p - 1.0).abs() < 1E-10);
    ///
    /// let sum_p_given = logp_swims_given_flippers
    ///     .iter()
    ///     .map(|lp| lp.exp())
    ///     .sum::<f64>();
    ///
    /// assert!((sum_p_given - 1.0).abs() < 1E-10);
    /// ```
    fn logp<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        vals: &[Vec<Datum>],
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<Vec<f64>, error::LogpError>;

    /// A version of `logp` where the likelihood are scaled by the column modes.
    ///
    /// The goal of this function is to create a notion of logp that is more
    /// standardized across rare variants. For example, if there is a class, A,
    /// of object with a highly variable distribution, we would have a hard time
    /// comparing the surprisal of class A variant to the surprisal of variants
    /// under a non-variable class B.
    ///
    /// That's a long way of saying that this is a hack and there's not any
    /// mathematical rigor behind it.
    ///
    /// # Notes
    /// - The contribution of each column is individually scaled based on the max
    ///   likelihood of the individual mixture model, then the geometric mean is
    ///   computed over column contributions.
    /// - Generating the cache is expensive, so if you plan on calling the
    ///   function with the same `col_ixs` and `given`, then you should really
    ///   pre-generate the cache.
    ///
    /// # Arguments
    /// - col_ixs: An d-length vector of the indices of the columns comprising
    ///   the data.
    /// - vals: An n-length vector of d-length vectors. The joint probability of
    ///   each of the n entries will be computed.
    /// - given: an optional set of observations on which to condition the
    ///   PMF/PDF
    /// - state_ixs_opt: An optional vector of the state indices to use for the
    ///   logp computation. If `None`, all states are used.
    /// - col_max_logps_opt: An optional cache of scaling constants. If not
    ///   supplied, the cache will be generated by the function.    
    ///
    ///  # Example
    ///
    ///  ```
    /// # use braid::examples::Example;
    /// use braid::{OracleT, Datum, Given};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let logp_scaled = oracle.logp_scaled(
    ///     &["swims"],
    ///     &[vec![Datum::Categorical(0)]],
    ///     &Given::Nothing,
    ///     None,
    ///     None,
    /// ).unwrap()[0];
    ///  ```
    ///
    /// Pre-computing the normalizing values
    ///  ```
    /// # use braid::examples::Example;
    /// use braid::{OracleT, Datum, Given, ColumnMaximumLogpCache};
    /// use braid::examples::animals::Column;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let col_ixs: [usize; 1] = [Column::Swims.into()];
    /// let given = Given::Nothing;
    /// let cache = ColumnMaximumLogpCache::from_oracle(
    ///     &oracle,
    ///     &col_ixs,
    ///     &given,
    ///     None
    /// );
    ///
    ///
    /// let logp_scaled = oracle.logp_scaled(
    ///     &col_ixs,
    ///     &[vec![Datum::Categorical(0)]],
    ///     &given,
    ///     None,
    ///     Some(&cache),
    /// ).unwrap()[0];
    ///  ```
    fn logp_scaled<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        vals: &[Vec<Datum>],
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
        col_max_logps_opt: Option<&ColumnMaximumLogpCache>,
    ) -> Result<Vec<f64>, error::LogpError>;

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
    /// use braid::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mut rng = rand::thread_rng();
    /// let xs = oracle.draw("pig", "fierce", 12, &mut rng).unwrap();
    ///
    /// assert_eq!(xs.len(), 12);
    /// assert!(xs.iter().all(|x| x.is_categorical()));
    /// ```
    fn draw<R: Rng, RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        n: usize,
        rng: &mut R,
    ) -> Result<Vec<Datum>, IndexError>;

    /// Simulate values from joint or conditional distribution
    ///
    /// # Arguments
    ///
    /// - col_ixs: a d-length vector containing the column indices to simulate
    /// - given: optional observations by which to constrain the simulation,
    ///   i.e., simulate from p(col_ixs|given)
    /// - n: the number of simulation
    /// - state_ixs_opt: The indices of the states from which to simulate. If
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
    /// use braid::OracleT;
    /// use braid::Given;
    /// use braid::examples::animals::Column;
    /// use braid_data::Datum;
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
    ///     &["black", "tail"],
    ///     &given,
    ///     10,
    ///     None,
    ///     &mut rng,
    /// ).unwrap();
    ///
    /// assert_eq!(xs.len(), 10);
    /// assert!(xs.iter().all(|x| x.len() == 2));
    /// ```
    fn simulate<R: Rng, Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        given: &Given,
        n: usize,
        state_ixs_opt: Option<Vec<usize>>,
        rng: &mut R,
    ) -> Result<Vec<Vec<Datum>>, error::SimulateError>;

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
    /// ```no_run
    /// # use braid::examples::Example;
    /// use braid::OracleT;
    /// use braid::ImputeUncertaintyType;
    /// use braid_data::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let dolphin_swims = oracle.impute(
    ///     "dolphin",
    ///     "swims",
    ///     Some(ImputeUncertaintyType::JsDivergence)
    /// ).unwrap();
    ///
    /// let bear_swims = oracle.impute(
    ///     "polar+bear",
    ///     "swims",
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
    fn impute<RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        unc_type_opt: Option<ImputeUncertaintyType>,
    ) -> Result<(Datum, Option<f64>), IndexError>;

    /// Return the most likely value for a column given a set of conditions
    /// along with the confidence in that prediction.
    ///
    /// # Arguments
    /// - col_ix: the index of the column to predict
    /// - given: optional observations by which to constrain the prediction
    ///
    /// # Returns
    /// A `(value, uncertainty_option)` Tuple
    fn predict<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
        given: &Given,
        unc_type_opt: Option<PredictUncertaintyType>,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<(Datum, Option<f64>), error::PredictError>;

    /// Compute the error between the observed data in a feature and the feature
    /// model.
    ///
    /// # Returns
    /// An `(error, centroid)` tuple where error a float in [0, 1], and the
    /// centroid is the centroid of  the error. For continuous features, the
    /// error is derived from the probability integral transform, and for
    /// discrete variables the error is the error between the inferred and
    /// empirical CDFs.
    fn feature_error<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
    ) -> Result<(f64, f64), IndexError>;

    // Private function impls
    // ---------------------
    fn _logp_unchecked(
        &self,
        col_ixs: &[usize],
        vals: &[Vec<Datum>],
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
        col_max_logps: Option<&ColumnMaximumLogpCache>,
    ) -> Vec<f64>;

    fn _simulate_unchecked<R: Rng>(
        &self,
        col_ixs: &[usize],
        given: &Given,
        n: usize,
        state_ixs_opt: Option<Vec<usize>>,
        rng: &mut R,
    ) -> Vec<Vec<Datum>>;

    fn _surprisal_unchecked(
        &self,
        x: &Datum,
        row_ix: usize,
        col_ix: usize,
        state_ixs_opt: Option<Vec<usize>>,
    ) -> Option<f64>;

    /// specialization for column pairs. If a specialization is not founds for
    /// the specific columns types, will fall back to MC approximation
    fn _dual_entropy(&self, col_a: usize, col_b: usize, n: usize) -> f64;

    /// Get the components of mutual information between two columns
    fn _mi_components(
        &self,
        col_a: usize,
        col_b: usize,
        n: usize,
    ) -> MiComponents;

    /// Use a Sobol QMC sequence to appropriate joint entropy
    ///
    /// # Notes
    /// This thing is shit. Don't use it.
    fn _sobol_joint_entropy(&self, col_ixs: &[usize], n: usize) -> f64;

    // Use Monte Carlo to estimate the joint entropy
    fn _mc_joint_entropy<R: Rng>(
        &self,
        col_ixs: &[usize],
        n: usize,
        rng: &mut R,
    ) -> f64;

    fn _entropy_unchecked(&self, col_ixs: &[usize], n: usize) -> f64;

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
    /// - unc_type: The type of uncertainty to compute
    fn _impute_uncertainty(
        &self,
        row_ix: usize,
        col_ix: usize,
        unc_type: ImputeUncertaintyType,
    ) -> f64;

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
    fn _predict_uncertainty(
        &self,
        col_ix: usize,
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
    ) -> f64;
}

fn extract_colix<Ix: Into<ColumnIndex>>(
    col_ix: Ix,
    codebook: &Codebook,
) -> Result<usize, IndexError> {
    use crate::NameOrIndex;

    let col_ix: ColumnIndex = col_ix.into();

    match col_ix.0 {
        NameOrIndex::Index(ix) => {
            let n_cols = codebook.n_cols();
            if ix < n_cols {
                Ok(ix)
            } else {
                Err(IndexError::ColumnIndexOutOfBounds { n_cols, col_ix: ix })
            }
        }
        NameOrIndex::Name(name) => codebook
            .column_index(&name)
            .ok_or(IndexError::ColumnNameDoesNotExist { name }),
    }
}

fn extract_colixs<Ix>(
    col_ixs: &[Ix],
    codebook: &Codebook,
) -> Result<Vec<usize>, IndexError>
where
    Ix: Into<ColumnIndex> + Clone,
{
    col_ixs
        .iter()
        .map(|col_ix| extract_colix(col_ix.clone(), codebook))
        .collect()
}

fn extract_col_pair<Ix: Into<ColumnIndex> + Clone>(
    pair: &(Ix, Ix),
    codebook: &Codebook,
) -> Result<(usize, usize), IndexError> {
    extract_colix(pair.0.clone(), codebook).and_then(|ix_a| {
        extract_colix(pair.1.clone(), codebook).map(|ix_b| (ix_a, ix_b))
    })
}

fn extract_rowix<Ix: Into<RowIndex>>(
    row_ix: Ix,
    codebook: &Codebook,
) -> Result<usize, IndexError> {
    use crate::NameOrIndex;

    let row_ix: RowIndex = row_ix.into();
    match row_ix.0 {
        NameOrIndex::Index(ix) => {
            let n_rows = codebook.row_names.len();
            if ix < n_rows {
                Ok(ix)
            } else {
                Err(IndexError::RowIndexOutOfBounds { n_rows, row_ix: ix })
            }
        }
        NameOrIndex::Name(name) => codebook
            .row_index(&name)
            .ok_or(IndexError::RowNameDoesNotExist { name }),
    }
}

fn extract_row_pair<Ix: Into<RowIndex> + Clone>(
    pair: &(Ix, Ix),
    codebook: &Codebook,
) -> Result<(usize, usize), IndexError> {
    extract_rowix(pair.0.clone(), codebook).and_then(|ix_a| {
        extract_rowix(pair.1.clone(), codebook).map(|ix_b| (ix_a, ix_b))
    })
}

impl<T: CanOracle> OracleT for T {
    fn state_diagnostics(&self) -> Vec<StateDiagnostics> {
        self.states()
            .iter()
            .map(|state| state.diagnostics.clone())
            .collect()
    }

    fn ftype<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
    ) -> Result<FType, IndexError> {
        let col_ix = extract_colix(col_ix, self.codebook())?;
        let state = &self.states()[0];
        let view_ix = state.asgn.asgn[col_ix];
        Ok(state.views[view_ix].ftrs[&col_ix].ftype())
    }

    fn depprob<Ix: Into<ColumnIndex>>(
        &self,
        col_a: Ix,
        col_b: Ix,
    ) -> Result<f64, IndexError> {
        let col_a = extract_colix(col_a, self.codebook())?;
        let col_b = extract_colix(col_b, self.codebook())?;

        if col_a == col_b {
            Ok(1.0)
        } else {
            let depprob = self.states().iter().fold(0.0, |acc, state| {
                if state.asgn.asgn[col_a] == state.asgn.asgn[col_b] {
                    acc + 1.0
                } else {
                    acc
                }
            }) / (self.n_states() as f64);
            Ok(depprob)
        }
    }

    fn rowsim<RIx: Into<RowIndex>, CIx: Into<ColumnIndex> + Clone>(
        &self,
        row_a: RIx,
        row_b: RIx,
        wrt: Option<&[CIx]>,
        col_weighted: bool,
    ) -> Result<f64, error::RowSimError> {
        let row_a = extract_rowix(row_a, self.codebook())?;
        let row_b = extract_rowix(row_b, self.codebook())?;
        let wrt = wrt
            .map(|col_ixs| extract_colixs(col_ixs, self.codebook()))
            .transpose()
            .map_err(error::RowSimError::WrtColumnIndexOutOfBounds)?;

        self._rowsim_validation(row_a, row_b, &wrt.as_ref())?;

        if row_a == row_b {
            return Ok(1.0);
        }

        let rowsim = self.states().iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt.as_ref() {
                Some(col_ixs) => {
                    let asgn = &state.asgn.asgn;
                    let viewset: BTreeSet<usize> =
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]).collect();
                    viewset.iter().copied().collect()
                }
                None => (0..state.views.len()).collect(),
            };

            let (norm, col_counts) = if col_weighted {
                let col_counts: Vec<f64> = view_ixs
                    .iter()
                    .map(|&ix| state.views[ix].n_cols() as f64)
                    .collect();
                (col_counts.iter().cloned().sum(), Some(col_counts))
            } else {
                (view_ixs.len() as f64, None)
            };

            acc + view_ixs.iter().enumerate().fold(
                0.0,
                |sim, (ix, &view_ix)| {
                    let asgn = &state.views[view_ix].asgn.asgn;
                    if asgn[row_a] == asgn[row_b] {
                        sim + col_counts.as_ref().map_or(1.0, |cts| cts[ix])
                    } else {
                        sim
                    }
                },
            ) / norm
        }) / self.n_states() as f64;

        Ok(rowsim)
    }

    fn novelty<RIx: Into<RowIndex>, CIx: Into<ColumnIndex> + Clone>(
        &self,
        row_ix: RIx,
        wrt: Option<&[CIx]>,
    ) -> Result<f64, IndexError> {
        let row_ix = extract_rowix(row_ix, self.codebook())?;
        let wrt = wrt
            .map(|col_ixs| extract_colixs(col_ixs, self.codebook()))
            .transpose()?;

        let nf = self.n_rows() as f64;

        let compliment = self.states().iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt.as_ref() {
                Some(col_ixs) => {
                    let asgn = &state.asgn.asgn;
                    let viewset: BTreeSet<usize> =
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]).collect();
                    viewset.iter().copied().collect()
                }
                None => (0..state.views.len()).collect(),
            };

            acc + view_ixs.iter().fold(0.0, |novelty, &view_ix| {
                let asgn = &state.views[view_ix].asgn;
                let z = asgn.asgn[row_ix];
                novelty + (asgn.counts[z] as f64) / nf
            }) / (view_ixs.len() as f64)
        }) / self.n_states() as f64;

        Ok(1.0 - compliment)
    }

    fn mi_pw<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_pairs: &[(Ix, Ix)],
        n: usize,
        mi_type: MiType,
    ) -> Result<Vec<f64>, error::MiError> {
        if col_pairs.is_empty() {
            return Ok(Vec::new());
        }

        // TODO: better to re-convert or allocate a new pairs vec?
        let col_pairs: Vec<(usize, usize)> = col_pairs
            .iter()
            .map(|(ix_a, ix_b)| {
                extract_colix(ix_a.clone(), self.codebook()).and_then(|a| {
                    extract_colix(ix_b.clone(), self.codebook()).map(|b| (a, b))
                })
            })
            .collect::<Result<Vec<(usize, usize)>, error::IndexError>>()?;

        // Precompute the single-column entropies
        let mut col_ixs: BTreeSet<usize> = BTreeSet::new();
        col_pairs.iter().for_each(|(col_a, col_b)| {
            col_ixs.insert(*col_a);
            col_ixs.insert(*col_b);
        });
        let max_ix = col_ixs.iter().max().unwrap();

        let n_cols = self.n_cols();
        col_indices_ok!(n_cols, col_ixs, IndexError::ColumnIndexOutOfBounds)?;

        let entropies = {
            let mut entropies = vec![0_f64; max_ix + 1];
            col_ixs.iter().for_each(|&col_ix| {
                let h = utils::entropy_single(col_ix, self.states());
                entropies[col_ix] = h;
            });
            entropies
        };

        let mis: Vec<_> = col_pairs
            .par_iter()
            .map(|&(col_a, col_b)| {
                let h_a = entropies[col_a];
                let mi_cpnts = if col_a == col_b {
                    // By definition, H(X, X) = H(X)
                    MiComponents {
                        h_a,
                        h_b: h_a,
                        h_ab: h_a,
                    }
                } else {
                    let h_b = entropies[col_b];
                    let h_ab = self._dual_entropy(col_a, col_b, n);
                    MiComponents { h_a, h_b, h_ab }
                };
                mi_cpnts.compute(mi_type)
            })
            .collect();

        Ok(mis)
    }

    fn conditional_entropy_pw<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_pairs: &[(Ix, Ix)],
        n: usize,
        kind: ConditionalEntropyType,
    ) -> Result<Vec<f64>, error::ConditionalEntropyError> {
        if col_pairs.is_empty() {
            return Ok(vec![]);
        } else if n == 0 {
            return Err(error::ConditionalEntropyError::NIsZero);
        };

        // TODO: better to re-convert or allocate a new pairs vec?
        let col_pairs: Vec<(usize, usize)> = col_pairs
            .iter()
            .map(|(ix_a, ix_b)| {
                extract_colix(ix_a.clone(), self.codebook())
                    .map_err(error::ConditionalEntropyError::TargetIndexOutOfBounds)
                    .and_then(|a| {
                        extract_colix(ix_b.clone(), self.codebook()).map(|b| (a, b))
                        .map_err(error::ConditionalEntropyError::PredictorIndexOutOfBounds)
                })
            })
            .collect::<Result<Vec<(usize, usize)>, error::ConditionalEntropyError>>()?;

        col_pairs
            .par_iter()
            .map(|&(col_a, col_b)| match kind {
                ConditionalEntropyType::InfoProp => {
                    let MiComponents { h_a, h_b, h_ab } =
                        self._mi_components(col_a, col_b, n);
                    Ok((h_a + h_b - h_ab) / h_a)
                }
                ConditionalEntropyType::UnNormed => {
                    let h_b = utils::entropy_single(col_b, self.states());
                    let h_ab = self._dual_entropy(col_a, col_b, n);
                    Ok(h_ab - h_b)
                }
            })
            .collect()
    }

    fn logp<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        vals: &[Vec<Datum>],
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<Vec<f64>, error::LogpError> {
        if col_ixs.is_empty() {
            return Err(error::LogpError::NoTargets);
        }

        let col_ixs = extract_colixs(col_ixs, self.codebook())
            .map_err(error::LogpError::TargetIndexOutOfBounds)?;

        find_given_errors(&col_ixs, &self.states()[0], given)
            .map_err(|err| err.into())
            .and_then(|_| {
                find_value_conflicts(&col_ixs, vals, &self.states()[0])
            })?;

        match state_ixs_opt {
            Some(state_ixs) if state_ixs.is_empty() => {
                Err(error::LogpError::NoStateIndices)
            }
            Some(state_ixs) => state_indices_ok!(
                self.n_states(),
                state_ixs,
                error::LogpError::StateIndexOutOfBounds
            ),
            None => Ok(()),
        }
        .map(|_| {
            self._logp_unchecked(&col_ixs, vals, given, state_ixs_opt, None)
        })
    }

    fn logp_scaled<Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        vals: &[Vec<Datum>],
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
        col_max_logps_opt: Option<&ColumnMaximumLogpCache>,
    ) -> Result<Vec<f64>, error::LogpError> {
        if col_ixs.is_empty() {
            return Err(error::LogpError::NoTargets);
        }

        let col_ixs = extract_colixs(col_ixs, self.codebook())
            .map_err(error::LogpError::TargetIndexOutOfBounds)?;

        find_given_errors(&col_ixs, &self.states()[0], given)
            .map_err(|err| err.into())
            .and_then(|_| {
                find_value_conflicts(&col_ixs, vals, &self.states()[0])
            })?;

        match state_ixs_opt {
            Some(state_ixs) if state_ixs.is_empty() => {
                Err(error::LogpError::NoStateIndices)
            }
            Some(state_ixs) => state_indices_ok!(
                self.n_states(),
                state_ixs,
                error::LogpError::StateIndexOutOfBounds
            ),
            None => Ok(()),
        }?;

        let res = col_max_logps_opt.map_or_else(
            || {
                let col_max_logps = ColumnMaximumLogpCache::from_oracle(
                    self,
                    &col_ixs,
                    given,
                    state_ixs_opt,
                );
                col_max_logps
                    .validate(self, &col_ixs, given, state_ixs_opt)
                    .map(|_| {
                        self._logp_unchecked(
                            &col_ixs,
                            vals,
                            given,
                            state_ixs_opt,
                            Some(&col_max_logps),
                        )
                    })
            },
            |col_max_logps| {
                col_max_logps
                    .validate(self, &col_ixs, given, state_ixs_opt)
                    .map(|_| {
                        self._logp_unchecked(
                            &col_ixs,
                            vals,
                            given,
                            state_ixs_opt,
                            Some(col_max_logps),
                        )
                    })
            },
        )?;

        Ok(res)
    }

    fn draw<R: Rng, RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        n: usize,
        mut rng: &mut R,
    ) -> Result<Vec<Datum>, IndexError> {
        let row_ix = extract_rowix(row_ix, self.codebook())?;
        let col_ix = extract_colix(col_ix, self.codebook())?;

        if n == 0 {
            return Ok(Vec::new());
        }

        let state_ixer = Categorical::uniform(self.n_states());
        let draws: Vec<_> = (0..n)
            .map(|_| {
                // choose a random state
                let state_ix: usize = state_ixer.draw(&mut rng);
                let state = &self.states()[state_ix];

                // Draw from the propoer component in the feature
                let view_ix = state.asgn.asgn[col_ix];
                let cpnt_ix = state.views[view_ix].asgn.asgn[row_ix];
                let ftr = state.feature(col_ix);
                ftr.draw(cpnt_ix, &mut rng)
            })
            .collect();
        Ok(draws)
    }

    fn simulate<R: Rng, Ix: Into<ColumnIndex> + Clone>(
        &self,
        col_ixs: &[Ix],
        given: &Given,
        n: usize,
        state_ixs_opt: Option<Vec<usize>>,
        mut rng: &mut R,
    ) -> Result<Vec<Vec<Datum>>, error::SimulateError> {
        if col_ixs.is_empty() {
            return Err(error::SimulateError::NoTargets);
        }

        let col_ixs = extract_colixs(col_ixs, self.codebook())
            .map_err(error::SimulateError::TargetIndexOutOfBounds)?;

        if let Some(ref state_ixs) = state_ixs_opt {
            if state_ixs.is_empty() {
                return Err(error::SimulateError::NoStateIndices);
            }
            state_indices_ok!(
                self.n_states(),
                state_ixs,
                error::SimulateError::StateIndexOutOfBounds
            )?;
        }

        find_given_errors(&col_ixs, &self.states()[0], given)?;

        Ok(self._simulate_unchecked(
            &col_ixs,
            given,
            n,
            state_ixs_opt,
            &mut rng,
        ))
    }

    fn impute<RIx: Into<RowIndex>, CIx: Into<ColumnIndex>>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        unc_type_opt: Option<ImputeUncertaintyType>,
    ) -> Result<(Datum, Option<f64>), IndexError> {
        let row_ix = extract_rowix(row_ix, self.codebook())?;
        let col_ix = extract_colix(col_ix, self.codebook())?;

        let states: Vec<&State> = self.states().iter().collect();
        let val: Datum = match self.ftype(col_ix).unwrap() {
            FType::Continuous => {
                let x = utils::continuous_impute(&states, row_ix, col_ix);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x = utils::categorical_impute(&states, row_ix, col_ix);
                Datum::Categorical(x)
            }
            FType::Labeler => {
                let x = utils::labeler_impute(&states, row_ix, col_ix);
                Datum::Label(x)
            }
            FType::Count => {
                let x = utils::count_impute(&states, row_ix, col_ix);
                Datum::Count(x)
            }
        };

        let unc_opt = unc_type_opt
            .map(|unc_type| self._impute_uncertainty(row_ix, col_ix, unc_type));

        Ok((val, unc_opt))
    }

    fn predict<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
        given: &Given,
        unc_type_opt: Option<PredictUncertaintyType>,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<(Datum, Option<f64>), error::PredictError> {
        let col_ix = extract_colix(col_ix, self.codebook())?;

        find_given_errors(&[col_ix], &self.states()[0], given)?;

        let states = utils::select_states(self.states(), state_ixs_opt);

        let value = match self.ftype(col_ix).unwrap() {
            FType::Continuous => {
                let x = utils::continuous_predict(&states, col_ix, given);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x = utils::categorical_predict(&states, col_ix, given);
                Datum::Categorical(x)
            }
            FType::Labeler => {
                let x = utils::labeler_predict(&states, col_ix, given);
                Datum::Label(x)
            }
            FType::Count => {
                let x = utils::count_predict(&states, col_ix, given);
                Datum::Count(x)
            }
        };

        let unc_opt = unc_type_opt
            .map(|_| self._predict_uncertainty(col_ix, given, state_ixs_opt));

        Ok((value, unc_opt))
    }

    fn feature_error<Ix: Into<ColumnIndex>>(
        &self,
        col_ix: Ix,
    ) -> Result<(f64, f64), IndexError> {
        let col_ix = extract_colix(col_ix, self.codebook())?;

        // extract the feature from the first state
        let ftr = self.states()[0].feature(col_ix);
        let ftype = ftr.ftype();

        let err = if ftype.is_continuous() {
            feature_err_arm!(self, col_ix, Gaussian, f64, |row_ix, col_ix| self
                .cell(row_ix, col_ix)
                .to_f64_opt())
        } else if ftype.is_categorical() {
            feature_err_arm!(self, col_ix, Categorical, u8, |row_ix, col_ix| {
                self.cell(row_ix, col_ix).to_u8_opt()
            })
        } else {
            panic!("Unsupported feature type");
        };

        Ok(err)
    }

    fn _logp_unchecked(
        &self,
        col_ixs: &[usize],
        vals: &[Vec<Datum>],
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
        col_max_logps: Option<&ColumnMaximumLogpCache>,
    ) -> Vec<f64> {
        let states = utils::select_states(self.states(), state_ixs_opt);
        let weights = utils::state_weights(&states, col_ixs, given);
        let mut vals_iter = vals.iter();

        let calculator = if let Some(cmls) = col_max_logps {
            utils::Calculator::new_scaled(
                &mut vals_iter,
                &states,
                &weights,
                col_ixs,
                &cmls.cache,
            )
        } else {
            utils::Calculator::new(&mut vals_iter, &states, &weights, col_ixs)
        };

        calculator.collect()
    }

    fn _simulate_unchecked<R: Rng>(
        &self,
        col_ixs: &[usize],
        given: &Given,
        n: usize,
        state_ixs_opt: Option<Vec<usize>>,
        mut rng: &mut R,
    ) -> Vec<Vec<Datum>> {
        let states: Vec<&State> = match state_ixs_opt {
            Some(ref state_ixs) => {
                state_ixs.iter().map(|&ix| &self.states()[ix]).collect()
            }
            None => self.states().iter().collect(),
        };

        let weights = utils::given_weights(&states, col_ixs, given);

        let simulator = utils::Simulator::new(
            &states,
            &weights,
            state_ixs_opt,
            col_ixs,
            &mut rng,
        );

        simulator.take(n).collect()
    }

    fn _surprisal_unchecked(
        &self,
        x: &Datum,
        row_ix: usize,
        col_ix: usize,
        state_ixs_opt: Option<Vec<usize>>,
    ) -> Option<f64> {
        if x.is_missing() {
            return None;
        }

        let state_ixs = state_ixs_opt
            .unwrap_or_else(|| (0..self.n_states()).collect::<Vec<_>>());

        let n_states = state_ixs.len();

        let logps: Vec<f64> = state_ixs
            .iter()
            .map(|&ix| {
                let state = &self.states()[ix];
                let view_ix = state.asgn.asgn[col_ix];
                let k = state.views[view_ix].asgn.asgn[row_ix];
                state.views[view_ix].ftrs[&col_ix].cpnt_logp(x, k)
            })
            .collect();
        let s = -logsumexp(&logps) + (n_states as f64).ln();
        Some(s)
    }

    fn _dual_entropy(&self, col_a: usize, col_b: usize, n: usize) -> f64 {
        let ftypes = (self.ftype(col_a).unwrap(), self.ftype(col_b).unwrap());
        match ftypes {
            (FType::Categorical, FType::Categorical) => {
                utils::categorical_entropy_dual(col_a, col_b, self.states())
            }
            (FType::Categorical, FType::Continuous) => {
                utils::categorical_gaussian_entropy_dual(
                    col_a,
                    col_b,
                    self.states(),
                )
            }
            (FType::Continuous, FType::Categorical) => {
                utils::categorical_gaussian_entropy_dual(
                    col_b,
                    col_a,
                    self.states(),
                )
            }
            (FType::Count, FType::Count) => {
                utils::count_entropy_dual(col_b, col_a, self.states())
            }
            _ => {
                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256Plus;
                let mut rng = Xoshiro256Plus::seed_from_u64(1337);
                self._mc_joint_entropy(&[col_a, col_b], n, &mut rng)
            }
        }
    }

    /// Get the components of mutual information between two columns
    fn _mi_components(
        &self,
        col_a: usize,
        col_b: usize,
        n: usize,
    ) -> MiComponents {
        let h_a = utils::entropy_single(col_a, self.states());
        if col_a == col_b {
            MiComponents {
                h_a,
                h_b: h_a,
                h_ab: h_a,
            }
        } else {
            let h_b = utils::entropy_single(col_b, self.states());
            let h_ab = self._dual_entropy(col_a, col_b, n);
            MiComponents { h_a, h_b, h_ab }
        }
    }

    /// Use a Sobol QMC sequence to appropriate joint entropy
    ///
    // # Note
    // This thing is shit. Don't use it.
    fn _sobol_joint_entropy(&self, col_ixs: &[usize], n: usize) -> f64 {
        let (vals, q_recip) =
            utils::gen_sobol_samples(col_ixs, &self.states()[0], n);
        let logps =
            self._logp_unchecked(col_ixs, &vals, &Given::Nothing, None, None);
        let h: f64 = logps.iter().map(|logp| -logp * logp.exp()).sum();
        h * q_recip / (n as f64)
    }

    /// Use Monte Carlo to estimate joint entropy
    ///
    /// # Arguments
    /// - col_ixs: the indices of the columns in the joint distribution
    /// - n: the number of sample to use for the Monte Carlo integral
    /// - rng: the random number generator
    fn _mc_joint_entropy<R: Rng>(
        &self,
        col_ixs: &[usize],
        n: usize,
        mut rng: &mut R,
    ) -> f64 {
        let states: Vec<_> = self.states().iter().collect();
        let weights = utils::given_weights(&states, col_ixs, &Given::Nothing);
        // Draws from p(x_1, x_2, ...)
        let mut simulator =
            utils::Simulator::new(&states, &weights, None, col_ixs, &mut rng);
        // Computes ln p (x_1, x_2, ...)
        let calculator =
            utils::Calculator::new(&mut simulator, &states, &weights, col_ixs);

        -calculator.take(n).sum::<f64>() / (n as f64)
    }

    fn _entropy_unchecked(&self, col_ixs: &[usize], n: usize) -> f64 {
        let all_categorical = col_ixs
            .iter()
            .all(|&ix| self.ftype(ix) == Ok(FType::Categorical));

        match col_ixs.len() {
            0 => unreachable!(),
            1 => utils::entropy_single(col_ixs[0], self.states()),
            2 => self._dual_entropy(col_ixs[0], col_ixs[1], n),
            _ if all_categorical => {
                utils::categorical_joint_entropy(col_ixs, self.states())
            }
            // _ => self.sobol_joint_entropy(col_ixs, n),
            _ => {
                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256Plus;
                let mut rng = Xoshiro256Plus::seed_from_u64(1337);
                self._mc_joint_entropy(col_ixs, n, &mut rng)
            }
        }
    }

    #[inline]
    fn _impute_uncertainty(
        &self,
        row_ix: usize,
        col_ix: usize,
        unc_type: ImputeUncertaintyType,
    ) -> f64 {
        match unc_type {
            ImputeUncertaintyType::JsDivergence => {
                utils::js_impute_uncertainty(self.states(), row_ix, col_ix)
            }
            ImputeUncertaintyType::PairwiseKl => {
                utils::kl_impute_uncertainty(self.states(), row_ix, col_ix)
            }
        }
    }

    #[inline]
    fn _predict_uncertainty(
        &self,
        col_ix: usize,
        given: &Given,
        state_ixs_opt: Option<&[usize]>,
    ) -> f64 {
        utils::predict_uncertainty(self.states(), col_ix, given, state_ixs_opt)
    }
}

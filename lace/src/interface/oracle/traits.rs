use super::error::{self, IndexError};
use super::validation::{find_given_errors, find_value_conflicts};
use super::{utils, RowSimilarityVariant};
use crate::index::{
    extract_col_pair, extract_colixs, extract_row_pair, ColumnIndex, RowIndex,
};
use crate::interface::oracle::error::SurprisalError;
use crate::interface::oracle::{ConditionalEntropyType, MiComponents, MiType};
use crate::interface::{CanOracle, Given};
use lace_cc::feature::{FType, Feature};
use lace_cc::state::{State, StateDiagnostics};
use lace_consts::rv::misc::LogSumExp;
use lace_data::{Datum, SummaryStatistics};
use lace_stats::rand;
use lace_stats::rand::Rng;
use lace_stats::rv::dist::{Categorical, Gaussian, Mixture};
use lace_stats::rv::traits::Sampleable;
use lace_stats::SampleError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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

/// Represents different formalizations of variability in distributions
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Variability {
    /// The variance of a univariate distribution
    Variance(f64),
    /// The entropy of a distribution
    Entropy(f64),
}

impl From<Variability> for f64 {
    fn from(value: Variability) -> Self {
        match value {
            Variability::Variance(x) => x,
            Variability::Entropy(x) => x,
        }
    }
}

pub trait OracleT: CanOracle {
    /// Returns the diagnostics for each state
    fn state_diagnostics(&self) -> Vec<StateDiagnostics> {
        self.states()
            .iter()
            .map(|state| state.diagnostics.clone())
            .collect()
    }

    /// Returns a tuple containing the number of rows, the number of columns,
    /// and the number of states
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let shape = oracle.shape();
    ///
    /// assert_eq!(shape, (50, 85, 16));
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
    /// # use lace::examples::Example;
    /// # use lace_cc::feature::FType;
    /// use lace::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let ftype = oracle.ftype("swims").unwrap();
    ///
    /// assert_eq!(ftype, FType::Categorical);
    /// ```
    fn ftype<Ix: ColumnIndex>(&self, col_ix: Ix) -> Result<FType, IndexError> {
        let col_ix = col_ix.col_ix(self.codebook())?;
        let state = &self.states()[0];
        let view_ix = state.asgn().asgn[col_ix];

        Ok(state.views[view_ix].ftrs[&col_ix].ftype())
    }

    /// Returns a vector of the feature types of each row
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
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

    /// Return a summary of the data in the column
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace_data::SummaryStatistics;
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
    fn summarize_col<Ix: ColumnIndex>(
        &self,
        col_ix: Ix,
    ) -> Result<SummaryStatistics, IndexError> {
        col_ix
            .col_ix(self.codebook())
            .map(|col_ix| self.summarize_feature(col_ix))
    }

    /// Estimated dependence probability between `col_a` and `col_b`
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
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
    fn depprob<Ix: ColumnIndex>(
        &self,
        col_a: Ix,
        col_b: Ix,
    ) -> Result<f64, IndexError> {
        let col_a = col_a.col_ix(self.codebook())?;
        let col_b = col_b.col_ix(self.codebook())?;

        if col_a == col_b {
            Ok(1.0)
        } else {
            let depprob = self.states().iter().fold(0.0, |acc, state| {
                if state.asgn().asgn[col_a] == state.asgn().asgn[col_b] {
                    acc + 1.0
                } else {
                    acc
                }
            }) / (self.n_states() as f64);
            Ok(depprob)
        }
    }

    /// Compute dependence probability for a list of column pairs.
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
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
        Ix: ColumnIndex,
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
    /// - variant: The type of row similarity to compute
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::RowSimilarityVariant;
    /// use lace::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let wrt: Option<&[usize]> = None;
    /// let rowsim = oracle.rowsim(
    ///     "wolf",
    ///     "collie",
    ///     wrt,
    ///     RowSimilarityVariant::ViewWeighted,
    /// ).unwrap();
    ///
    /// assert!(rowsim >= 0.0 && rowsim <= 1.0);
    /// ```
    /// Adding context with `wrt` (with respect to):
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace::RowSimilarityVariant;
    /// # let oracle = Example::Animals.oracle().unwrap();
    /// # let wrt: Option<&[usize]> = None;
    /// # let rowsim = oracle.rowsim(
    /// #     "wolf",
    /// #     "collie",
    /// #     wrt,
    /// #    RowSimilarityVariant::ViewWeighted,
    /// # ).unwrap();
    ///
    /// let rowsim_wrt = oracle.rowsim(
    ///     "wolf",
    ///     "collie",
    ///     Some(&["swims"]),
    ///     RowSimilarityVariant::ViewWeighted,
    /// ).unwrap();
    ///
    /// assert_ne!(rowsim, rowsim_wrt);
    /// ```
    fn rowsim<RIx: RowIndex, CIx: ColumnIndex>(
        &self,
        row_a: RIx,
        row_b: RIx,
        wrt: Option<&[CIx]>,
        variant: RowSimilarityVariant,
    ) -> Result<f64, error::RowSimError> {
        let row_a = row_a.row_ix(self.codebook())?;
        let row_b = row_b.row_ix(self.codebook())?;
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
                    let asgn = &state.asgn().asgn;
                    let viewset: BTreeSet<usize> =
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]).collect();
                    viewset.iter().copied().collect()
                }
                None => (0..state.views.len()).collect(),
            };

            let (norm, col_counts) = match variant {
                RowSimilarityVariant::ViewWeighted => {
                    (view_ixs.len() as f64, None)
                }
                RowSimilarityVariant::ColumnWeighted => {
                    let col_counts: Vec<f64> = view_ixs
                        .iter()
                        .map(|&ix| state.views[ix].n_cols() as f64)
                        .collect();
                    (col_counts.iter().cloned().sum(), Some(col_counts))
                }
            };

            acc + view_ixs.iter().enumerate().fold(
                0.0,
                |sim, (ix, &view_ix)| {
                    let asgn = &state.views[view_ix].asgn().asgn;
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

    /// Compute row similarity for pairs of rows
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::RowSimilarityVariant;
    /// use lace::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    /// let wrt: Option<&[usize]> = None;
    /// let rowsims = oracle.rowsim_pw(
    ///     &[
    ///         ("gorilla", "spider+monkey"),
    ///         ("gorilla", "skunk"),
    ///     ],
    ///     wrt,
    ///     RowSimilarityVariant::ViewWeighted,
    /// ).unwrap();
    ///
    /// assert!(rowsims.iter().all(|&rowsim| 0.0 <= rowsim && rowsim <= 1.0));
    /// ```
    fn rowsim_pw<'x, RIx, CIx>(
        &self,
        pairs: &'x [(RIx, RIx)],
        wrt: Option<&[CIx]>,
        variant: RowSimilarityVariant,
    ) -> Result<Vec<f64>, error::RowSimError>
    where
        RIx: RowIndex,
        CIx: ColumnIndex + Sync,
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
                        self.rowsim(row_a, row_b, wrt, variant)
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
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
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let wrt = vec!["swims"];
    ///
    /// let novelty_rat = oracle.novelty("rat", Some(&wrt)).unwrap();
    /// let novelty_dolphin = oracle.novelty("dolphin", Some(&wrt)).unwrap();
    ///
    /// assert!(novelty_dolphin > novelty_rat);
    /// ```
    fn novelty<RIx: RowIndex, CIx: ColumnIndex>(
        &self,
        row_ix: RIx,
        wrt: Option<&[CIx]>,
    ) -> Result<f64, IndexError> {
        let row_ix = row_ix.row_ix(self.codebook())?;
        let wrt = wrt
            .map(|col_ixs| extract_colixs(col_ixs, self.codebook()))
            .transpose()?;

        let nf = self.n_rows() as f64;

        let compliment = self.states().iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt.as_ref() {
                Some(col_ixs) => {
                    let asgn = &state.asgn().asgn;
                    let viewset: BTreeSet<usize> =
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]).collect();
                    viewset.iter().copied().collect()
                }
                None => (0..state.views.len()).collect(),
            };

            acc + view_ixs.iter().fold(0.0, |novelty, &view_ix| {
                let asgn = &state.views[view_ix].asgn();
                let z = asgn.asgn[row_ix];
                novelty + (asgn.counts[z] as f64) / nf
            }) / (view_ixs.len() as f64)
        }) / self.n_states() as f64;

        Ok(1.0 - compliment)
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace::MiType;
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
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace::MiType;
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
    ///
    /// Mutual information is not as well behaved for continuous variables since
    /// differential (continuous) entropy can be negative. The `Linfoot`
    /// `MiType` can help. Linfoot is a transformed mutual information variant
    /// that will be in the interval (0, 1).
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::prelude::*;
    /// let oracle = Example::Satellites.oracle().unwrap();
    ///
    /// let linfoot = oracle.mi(
    ///     "longitude_radians_of_geo",
    ///     "Eccentricity",
    ///     1000,
    ///     MiType::Linfoot,
    /// ).unwrap();
    ///
    /// assert!(0.0 < linfoot && linfoot < 1.0);
    /// ```
    fn mi<Ix: ColumnIndex>(
        &self,
        col_a: Ix,
        col_b: Ix,
        n: usize,
        mi_type: MiType,
    ) -> Result<f64, error::MiError> {
        if n == 0 {
            return Err(error::MiError::NIsZero);
        }

        let col_a = col_a.col_ix(self.codebook())?;
        let col_b = col_b.col_ix(self.codebook())?;

        let mi_cpnts = self._mi_components(col_a, col_b, n);
        Ok(mi_cpnts.compute(mi_type))
    }

    /// Compute mutual information over pairs of columns
    ///
    /// # Notes
    ///
    /// This function has special optimizations over computing oracle::mi for
    /// pairs manually.
    fn mi_pw<Ix: ColumnIndex>(
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
            .map(|pair| extract_col_pair(pair, self.codebook()))
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace::MiType;
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
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace::MiType;
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
    fn entropy<Ix: ColumnIndex>(
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
    /// # use lace::examples::Example;
    /// # use lace_cc::feature::FType;
    /// use lace::examples::animals::Column;
    /// use lace::OracleT;
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
    fn predictor_search<Ix: ColumnIndex>(
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
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
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
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
    fn info_prop<IxT: ColumnIndex, IxX: ColumnIndex>(
        &self,
        cols_t: &[IxT],
        cols_x: &[IxX],
        n: usize,
    ) -> Result<f64, error::InfoPropError> {
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace::examples::animals::Column;
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
    fn conditional_entropy<IxT: ColumnIndex, IxX: ColumnIndex>(
        &self,
        col_t: IxT,
        cols_x: &[IxX],
        n: usize,
    ) -> Result<f64, error::ConditionalEntropyError> {
        let col_t = col_t
            .col_ix(self.codebook())
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace::ConditionalEntropyType;
    /// use lace::examples::animals::Column;
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
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace::examples::animals::Column;
    /// # use lace::ConditionalEntropyType;
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
    fn conditional_entropy_pw<Ix: ColumnIndex>(
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
                ix_a.col_ix(self.codebook())
                    .map_err(error::ConditionalEntropyError::TargetIndexOutOfBounds)
                    .and_then(|a| {
                        ix_b.col_ix(self.codebook()).map(|b| (a, b))
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace_data::Datum;
    /// use lace::examples::animals::{Column, Row};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let present = Datum::Categorical(1_u32.into());
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
    fn surprisal<RIx: RowIndex, CIx: ColumnIndex>(
        &self,
        x: &Datum,
        row_ix: RIx,
        col_ix: CIx,
        state_ixs: Option<Vec<usize>>,
    ) -> Result<Option<f64>, error::SurprisalError> {
        let row_ix = row_ix.row_ix(self.codebook())?;
        let col_ix = col_ix.col_ix(self.codebook())?;

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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
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
    fn self_surprisal<RIx: RowIndex, CIx: ColumnIndex>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        state_ixs: Option<Vec<usize>>,
    ) -> Result<Option<f64>, error::SurprisalError> {
        let row_ix = row_ix.row_ix(self.codebook())?;
        let col_ix = col_ix.col_ix(self.codebook())?;

        self.datum(row_ix, col_ix)
            .map_err(SurprisalError::from)
            .map(|x| self._surprisal_unchecked(&x, row_ix, col_ix, state_ixs))
    }

    /// Get the datum at an index
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace_data::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let x = oracle.datum("pig", "fierce").unwrap();
    ///
    /// assert_eq!(x, Datum::Categorical(1_u32.into()));
    /// ```
    ///
    /// Getting data from the satellites dataset
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace_data::Datum;
    /// let oracle = Example::Satellites.oracle().unwrap();
    ///
    /// let x = oracle.datum(
    ///     "International Space Station (ISS [first element Zarya])",
    ///     "Class_of_Orbit"
    /// ).unwrap();
    ///
    /// assert_eq!(x, Datum::Categorical("LEO".into()));
    ///
    /// let y = oracle.datum(
    ///     "International Space Station (ISS [first element Zarya])",
    ///     "Period_minutes",
    /// ).unwrap();
    ///
    /// assert_eq!(y, Datum::Continuous(92.8));
    /// ```
    fn datum<RIx: RowIndex, CIx: ColumnIndex>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
    ) -> Result<Datum, IndexError> {
        let row_ix = row_ix.row_ix(self.codebook())?;
        let col_ix = col_ix.col_ix(self.codebook())?;

        let x = self.cell(row_ix, col_ix);
        Ok(utils::post_process_datum(x, col_ix, self.codebook()))
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace_data::Datum;
    /// use lace::Given;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let logp_swims = oracle.logp(
    ///     &["swims"],
    ///     &[vec![Datum::Categorical(0_u32.into())], vec![Datum::Categorical(1_u32.into())]],
    ///     &Given::<usize>::Nothing,
    ///     None,
    /// ).unwrap();
    ///
    /// let logp_swims_given_flippers = oracle.logp(
    ///     &["swims"],
    ///     &[
    ///         vec![Datum::Categorical(0_u32.into())],
    ///         vec![Datum::Categorical(1_u32.into())]
    ///     ],
    ///     &Given::Conditions(
    ///         vec![("flippers", Datum::Categorical(1_u32.into()))]
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
    ///
    /// For missing not at random columns, you can ask about the likelihood of
    /// missing values.
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace_data::Datum;
    /// # use lace::Given;
    /// let oracle = Example::Satellites.oracle().unwrap();
    ///
    /// let logps = oracle.logp(
    ///     &["longitude_radians_of_geo", "Type_of_Orbit", "Period_minutes"],
    ///     &[
    ///         vec![Datum::Missing, Datum::Missing, Datum::Continuous(70.0)],
    ///         vec![Datum::Missing, Datum::Categorical("Polar".into()), Datum::Continuous(70.0)],
    ///         vec![Datum::Continuous(1.2), Datum::Missing, Datum::Continuous(70.0)],
    ///     ],
    ///     &Given::<usize>::Nothing,
    ///     None,
    /// ).unwrap();
    ///
    /// assert!(logps[0] > logps[1]);
    /// assert!(logps[0] > logps[2]);
    /// ```
    ///
    /// And you can condition on missingness
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace_data::Datum;
    /// # use lace::Given;
    /// let oracle = Example::Satellites.oracle().unwrap();
    ///
    /// let logp = oracle.logp(
    ///     &["Period_minutes"],
    ///     &[
    ///         vec![Datum::Continuous(70.0)],   // ~LEO
    ///         vec![Datum::Continuous(300.0)],  // ~MEO
    ///         vec![Datum::Continuous(1440.0)], // ~GEO
    ///     ],
    ///     &Given::<usize>::Nothing,
    ///     None,
    /// ).unwrap();
    ///
    /// let logp_missing = oracle.logp(
    ///     &["Period_minutes"],
    ///     &[
    ///         vec![Datum::Continuous(70.0)],   // ~LEO
    ///         vec![Datum::Continuous(300.0)],  // ~MEO
    ///         vec![Datum::Continuous(1440.0)], // ~GEO
    ///     ],
    ///     &Given::Conditions(vec![
    ///         ("longitude_radians_of_geo", Datum::Missing)
    ///     ]),
    ///     None,
    /// ).unwrap();
    ///
    /// // LEO is more likely if no 'longitude_radians_of_geo' was given
    /// assert!(logp_missing[0] > logp[0]); // p LEO goes up w/ missing
    /// // GEO is less likely if no 'longitude_radians_of_geo' was given
    /// assert!(logp_missing[2] < logp[2]); // p GEO goes down
    /// ```
    fn logp<Ix: ColumnIndex, GIx: ColumnIndex>(
        &self,
        col_ixs: &[Ix],
        vals: &[Vec<Datum>],
        given: &Given<GIx>,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<Vec<f64>, error::LogpError> {
        if col_ixs.is_empty() {
            return Err(error::LogpError::NoTargets);
        }

        let col_ixs = extract_colixs(col_ixs, self.codebook())
            .map_err(error::LogpError::TargetIndexOutOfBounds)?;

        // TODO: determine with benchmarks whether it is better not to
        // canonicalize the given
        let given =
            given.clone().canonical(self.codebook()).map_err(|err| {
                error::LogpError::GivenError(error::GivenError::IndexError(err))
            })?;

        find_given_errors(&col_ixs, &self.states()[0], &given)
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
            self._logp_unchecked(&col_ixs, vals, &given, state_ixs_opt, false)
        })
    }

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
    ///
    ///  # Example
    ///
    ///  ```
    /// # use lace::examples::Example;
    /// use lace::{OracleT, Datum, Given};
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let logp_scaled = oracle.logp_scaled(
    ///     &["swims"],
    ///     &[vec![Datum::Categorical(0_u32.into())]],
    ///     &Given::<usize>::Nothing,
    ///     None,
    /// ).unwrap()[0];
    ///  ```
    fn logp_scaled<Ix: ColumnIndex, GIx: ColumnIndex>(
        &self,
        col_ixs: &[Ix],
        vals: &[Vec<Datum>],
        given: &Given<GIx>,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<Vec<f64>, error::LogpError>
    where
        Self: Sized,
    {
        if col_ixs.is_empty() {
            return Err(error::LogpError::NoTargets);
        }

        let col_ixs = extract_colixs(col_ixs, self.codebook())
            .map_err(error::LogpError::TargetIndexOutOfBounds)?;

        // TODO: determine with benchmarks whether it is better not to
        // canonicalize the given
        let given =
            given.clone().canonical(self.codebook()).map_err(|err| {
                error::LogpError::GivenError(error::GivenError::IndexError(err))
            })?;
        find_given_errors(&col_ixs, &self.states()[0], &given)
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

        Ok(self._logp_unchecked(&col_ixs, vals, &given, state_ixs_opt, true))
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mut rng = rand::thread_rng();
    /// let xs = oracle.draw("pig", "fierce", 12, &mut rng).unwrap();
    ///
    /// assert_eq!(xs.len(), 12);
    /// assert!(xs.iter().all(|x| x.is_categorical()));
    /// ```
    fn draw<RIx: RowIndex, CIx: ColumnIndex, R: Rng>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        n: usize,
        mut rng: &mut R,
    ) -> Result<Vec<Datum>, IndexError> {
        let row_ix = row_ix.row_ix(self.codebook())?;
        let col_ix = col_ix.col_ix(self.codebook())?;

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
                let view_ix = state.asgn().asgn[col_ix];
                let cpnt_ix = state.views[view_ix].asgn().asgn[row_ix];
                let ftr = state.feature(col_ix);
                let x = ftr.draw(cpnt_ix, &mut rng);
                utils::post_process_datum(x, col_ix, self.codebook())
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
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace::Given;
    /// use lace_data::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let given = Given::Conditions(
    ///     vec![
    ///         ("fierce", Datum::Categorical(1_u32.into())),
    ///         ("fast", Datum::Categorical(1_u32.into())),
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
    fn simulate<Ix: ColumnIndex, GIx: ColumnIndex, R: Rng>(
        &self,
        col_ixs: &[Ix],
        given: &Given<GIx>,
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

        // TODO: determine with benchmarks whether it is better not to
        // canonicalize the given
        let given =
            given.clone().canonical(self.codebook()).map_err(|err| {
                error::SimulateError::GivenError(error::GivenError::IndexError(
                    err,
                ))
            })?;
        find_given_errors(&col_ixs, &self.states()[0], &given)?;

        Ok(self._simulate_unchecked(
            &col_ixs,
            &given,
            n,
            state_ixs_opt,
            &mut rng,
        ))
    }

    /// Return the most likely value for a cell in the table along with the
    /// confidence in that imputation.
    ///
    /// Imputation can be done on non-missing cells and will re-predict the
    /// value of the cell rather than returning the existing value. To get
    /// the current value of a cell, use `Oracle::data`.
    ///
    /// Impute uncertainty is the mean [total variation distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures)
    /// between each state's impute distribution and the average impute
    /// distribution.
    ///
    /// # Arguments
    ///
    /// - row_ix: the row index of the cell to impute
    /// - col_ix: the column index of the cell to impute
    /// - with_uncertainty: if `true` compute and return the uncertainty
    ///
    /// # Returns
    ///
    /// A `(value, uncertainty_option)` tuple.
    ///
    /// # Example
    ///
    /// Impute the value of swims for an dolphin and an polar bear.
    ///
    /// ```
    /// # use lace::examples::Example;
    /// use lace::OracleT;
    /// use lace_data::Datum;
    ///
    /// let oracle = Example::Animals.oracle().unwrap();
    ///
    /// let dolphin_swims = oracle.impute(
    ///     "dolphin",
    ///     "swims",
    ///     true,
    /// ).unwrap();
    ///
    /// let bear_swims = oracle.impute(
    ///     "polar+bear",
    ///     "swims",
    ///     true,
    /// ).unwrap();
    ///
    /// assert_eq!(dolphin_swims.0, Datum::Categorical(1_u32.into()));
    /// assert_eq!(bear_swims.0, Datum::Categorical(1_u32.into()));
    ///
    /// let dolphin_swims_unc = dolphin_swims.1.unwrap();
    /// let bear_swims_unc = bear_swims.1.unwrap();
    ///
    /// // Given that a polar bear is a furry, footed mammal, it's harder to
    /// // model  why we know it swims.
    /// assert!(bear_swims_unc > dolphin_swims_unc);
    /// ```
    ///
    /// Imputing a missing-not-at-random value will still return a value
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace_data::{Datum, Category};
    /// let oracle = Example::Satellites.oracle().unwrap();
    ///
    /// let (imp, _) = oracle.impute(
    ///     "X-Sat",
    ///     "Type_of_Orbit",
    ///     true,
    /// ).unwrap();
    ///
    /// assert_eq!(imp, Datum::Categorical("Sun-Synchronous".into()));
    /// ```
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::OracleT;
    /// # use lace_data::{Datum, Category};
    /// # let oracle = Example::Satellites.oracle().unwrap();
    /// let (imp, unc): (Datum, Option<f64>) = oracle.impute(
    ///     "X-Sat",
    ///     "longitude_radians_of_geo",
    ///     true,
    /// ).unwrap();
    /// ```
    fn impute<RIx: RowIndex, CIx: ColumnIndex>(
        &self,
        row_ix: RIx,
        col_ix: CIx,
        with_uncertainty: bool,
    ) -> Result<(Datum, Option<f64>), IndexError> {
        let row_ix = row_ix.row_ix(self.codebook())?;
        let col_ix = col_ix.col_ix(self.codebook())?;

        let states: Vec<&State> = self.states().iter().collect();
        let val: Datum = match self.ftype(col_ix).unwrap() {
            FType::Binary => unimplemented!(),
            FType::Continuous => {
                let x = utils::continuous_impute(&states, row_ix, col_ix);
                Datum::Continuous(x)
            }
            FType::Categorical => {
                let x = utils::categorical_impute(&states, row_ix, col_ix);
                let cat =
                    utils::u32_to_category(x, col_ix, self.codebook()).unwrap();
                Datum::Categorical(cat)
            }
            FType::Count => {
                let x = utils::count_impute(&states, row_ix, col_ix);
                Datum::Count(x)
            }
        };

        let val = utils::post_process_datum(val, col_ix, self.codebook());

        let unc_opt = if with_uncertainty {
            Some(self._impute_uncertainty(row_ix, col_ix))
        } else {
            None
        };

        Ok((val, unc_opt))
    }

    /// Return the most likely value for a column given a set of conditions
    /// along with the confidence in that prediction.
    ///
    /// # Arguments
    /// - col_ix: the index of the column to predict
    /// - given: optional observations by which to constrain the prediction
    /// - with_uncertainty: if true, copmute and return uncertainty
    /// - state_ixs_opt: Optional vector of state indices from which to predict,
    ///   if None, use all states.
    ///
    /// # Returns
    /// A `(value, uncertainty_option)` Tuple
    ///
    /// # Examples
    ///
    /// Predict the most likely class of orbit for given longitude of
    /// Geosynchronous orbit.
    ///    
    /// ```
    /// use lace::examples::Example;
    /// use lace::prelude::*;
    ///
    /// let oracle = Example::Satellites.oracle().unwrap();
    ///
    /// let (pred, _) = oracle.predict(
    ///     "Class_of_Orbit",
    ///     &Given::Conditions(vec![
    ///         ("longitude_radians_of_geo", Datum::Continuous(1.0))
    ///     ]),
    ///     false,
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(pred, Datum::Categorical("GEO".into()));
    /// ```
    ///
    /// Predict the most likely class of orbit given the
    /// `longitude_radians_of_geo` field is missing. Note: this requires the
    /// column to be missing-not-at-random.
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::prelude::*;
    /// # let oracle = Example::Satellites.oracle().unwrap();
    /// let (pred_long_missing, _) = oracle.predict(
    ///     "Class_of_Orbit",
    ///     &Given::Conditions(vec![
    ///         ("longitude_radians_of_geo", Datum::Missing)
    ///     ]),
    ///     false,
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(pred_long_missing, Datum::Categorical("LEO".into()));
    /// ```
    ///
    /// Predict a categorical value that is missing not at random
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::prelude::*;
    /// # let oracle = Example::Satellites.oracle().unwrap();
    /// let (pred_type, _) = oracle.predict(
    ///     "Type_of_Orbit",
    ///     &Given::Conditions(vec![(
    ///         "Class_of_Orbit", Datum::Categorical("MEO".into()))
    ///     ]),
    ///     false,
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(pred_type, Datum::Missing);
    /// ```
    ///
    /// Predict a continuous value that is missing not at random and is missing
    /// most of the time
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::prelude::*;
    /// # let oracle = Example::Satellites.oracle().unwrap();
    /// let (pred_type, _) = oracle.predict(
    ///     "longitude_radians_of_geo",
    ///     &Given::<usize>::Nothing,
    ///     false,
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(pred_type, Datum::Missing);
    /// ```
    ///
    /// Note that the uncertainty when the prediction is missing is the
    /// uncertainty only of the missing prediction. For example, the
    /// `longitude_radians_of_geo` value is only present for geosynchronous
    /// satellites, which have an orbital period of around 1440 minutes. We can
    /// see the uncertainty drop as we condition on periods farther away from
    /// 1440 miuntues.
    ///
    /// ```
    /// # use lace::examples::Example;
    /// # use lace::prelude::*;
    /// # let oracle = Example::Satellites.oracle().unwrap();
    /// let (pred_close, unc_close) = oracle.predict(
    ///     "longitude_radians_of_geo",
    ///     &Given::Conditions(vec![
    ///         ("Period_minutes", Datum::Continuous(1400.0))
    ///     ]),
    ///     true,
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(pred_close, Datum::Missing);
    ///
    /// let (pred_far, unc_far) = oracle.predict(
    ///     "longitude_radians_of_geo",
    ///     &Given::Conditions(vec![
    ///         ("Period_minutes", Datum::Continuous(1000.0))
    ///     ]),
    ///     true,
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(pred_far, Datum::Missing);
    /// assert!(unc_far.unwrap() < unc_close.unwrap());
    /// ```
    fn predict<Ix: ColumnIndex, GIx: ColumnIndex>(
        &self,
        col_ix: Ix,
        given: &Given<GIx>,
        with_uncertainty: bool,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<(Datum, Option<f64>), error::PredictError> {
        use super::validation::Mnar;
        let col_ix = col_ix.col_ix(self.codebook())?;

        // TODO: determine with benchmarks whether it is better not to
        // canonicalize the given
        let given =
            given.clone().canonical(self.codebook()).map_err(|err| {
                error::PredictError::GivenError(error::GivenError::IndexError(
                    err,
                ))
            })?;
        find_given_errors(&[col_ix], &self.states()[0], &given)?;

        let states = utils::select_states(self.states(), state_ixs_opt);

        let is_mnar = states[0].feature(col_ix).is_mnar();

        let is_missing = if is_mnar {
            let p_missing = self._logp_unchecked(
                &[col_ix],
                &[vec![Datum::Missing]],
                &given,
                state_ixs_opt,
                false,
            )[0]
            .exp();
            p_missing > 0.5
        } else {
            false
        };
        if is_missing {
            let unc_opt = if with_uncertainty {
                Some(utils::mnar_uncertainty(states.as_slice(), col_ix, &given))
            } else {
                None
            };
            Ok((Datum::Missing, unc_opt))
        } else {
            let value = match self.ftype(col_ix).unwrap() {
                FType::Binary => unimplemented!(),
                FType::Continuous => {
                    let x = utils::continuous_predict(&states, col_ix, &given);
                    Datum::Continuous(x)
                }
                FType::Categorical => {
                    let x = utils::categorical_predict(&states, col_ix, &given);
                    let cat =
                        utils::u32_to_category(x, col_ix, self.codebook())
                            .unwrap();
                    Datum::Categorical(cat)
                }
                FType::Count => {
                    let x = utils::count_predict(&states, col_ix, &given);
                    Datum::Count(x)
                }
            };

            let value =
                utils::post_process_datum(value, col_ix, self.codebook());

            let unc_opt = if with_uncertainty {
                Some(self._predict_uncertainty(col_ix, &given, state_ixs_opt))
            } else {
                None
            };

            Ok((value, unc_opt))
        }
    }

    /// Compute the variability of a conditional distribution
    ///
    /// # Notes
    /// - Returns variance for Continuous and Count columns
    /// - Returns Entropy for Categorical columns
    ///
    /// # Arguments
    /// - col_ix: the index of the column for which to compute the variability
    /// - given: optional observations by which to constrain the prediction
    /// - state_ixs_opt: Optional vector of state indices from which to compute,
    ///   if None, use all states.
    fn variability<Ix: ColumnIndex, GIx: ColumnIndex>(
        &self,
        col_ix: Ix,
        given: &Given<GIx>,
        state_ixs_opt: Option<&[usize]>,
    ) -> Result<Variability, error::VariabilityError> {
        use crate::stats::rv::traits::{Entropy, Variance};
        use crate::stats::MixtureType;

        let states: Vec<&State> = if let Some(state_ixs) = state_ixs_opt {
            state_ixs.iter().map(|&ix| &self.states()[ix]).collect()
        } else {
            self.states().iter().collect()
        };

        let given =
            given.clone().canonical(self.codebook()).map_err(|err| {
                error::VariabilityError::GivenError(
                    error::GivenError::IndexError(err),
                )
            })?;

        let col_ix = col_ix.col_ix(self.codebook())?;

        // Get the mixture weights for each state
        let mut mixture_types: Vec<MixtureType> = states
            .iter()
            .map(|state| {
                let view_ix = state.asgn().asgn[col_ix];
                let weights =
                    &utils::given_weights(&[state], &[col_ix], &given)[0];

                // combine the state weights with the given weights
                let mut mm_weights: Vec<f64> = state.views[view_ix]
                    .weights
                    .iter()
                    .zip(weights[&view_ix].iter())
                    .map(|(&w1, &w2)| w1 + w2)
                    .collect();

                let z: f64 = mm_weights.iter().logsumexp();
                mm_weights.iter_mut().for_each(|w| *w = (*w - z).exp());

                state.views[view_ix].ftrs[&col_ix].to_mixture(mm_weights)
            })
            .collect();

        enum MType {
            Gaussian,
            Categorical,
            Count,
            Unsupported,
        }

        let mtype = match mixture_types[0] {
            MixtureType::Gaussian(_) => MType::Gaussian,
            MixtureType::Poisson(_) => MType::Count,
            MixtureType::Categorical(_) => MType::Categorical,
            _ => MType::Unsupported,
        };

        match mtype {
            MType::Gaussian => {
                let mms: Vec<_> = mixture_types
                    .drain(..)
                    .map(|mt| {
                        if let MixtureType::Gaussian(mm) = mt {
                            mm
                        } else {
                            panic!("Expected Gaussian Mixture Type")
                        }
                    })
                    .collect();
                let mm = Mixture::combine(mms);
                Ok(Variability::Variance(mm.variance().unwrap()))
            }
            MType::Count => {
                let mms: Vec<_> = mixture_types
                    .drain(..)
                    .map(|mt| {
                        if let MixtureType::Poisson(mm) = mt {
                            mm
                        } else {
                            panic!("Expected Poisson Mixture Type")
                        }
                    })
                    .collect();
                let mm = Mixture::combine(mms);
                Ok(Variability::Variance(mm.variance().unwrap()))
            }
            MType::Categorical => {
                let mms: Vec<_> = mixture_types
                    .drain(..)
                    .map(|mt| {
                        if let MixtureType::Categorical(mm) = mt {
                            mm
                        } else {
                            panic!("Expected Categorical Mixture Type")
                        }
                    })
                    .collect();
                let mm = Mixture::combine(mms);
                Ok(Variability::Entropy(mm.entropy()))
            }
            _ => panic!("Unsupported MType"),
        }
    }

    /// Compute the error between the observed data in a feature and the feature
    /// model.
    ///
    /// # Returns
    /// An `(error, centroid)` tuple where error a float in [0, 1], and the
    /// centroid is the centroid of  the error. For continuous features, the
    /// error is derived from the probability integral transform, and for
    /// discrete variables the error is the error between the inferred and
    /// empirical CDFs.
    #[allow(clippy::redundant_closure_call)]
    fn feature_error<Ix: ColumnIndex>(
        &self,
        col_ix: Ix,
    ) -> Result<(f64, f64), IndexError> {
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
        let col_ix = col_ix.col_ix(self.codebook())?;

        // extract the feature from the first state
        let ftr = self.states()[0].feature(col_ix);
        let ftype = ftr.ftype();

        let err = if ftype.is_continuous() {
            feature_err_arm!(self, col_ix, Gaussian, f64, |row_ix, col_ix| self
                .cell(row_ix, col_ix)
                .to_f64_opt())
        } else if ftype.is_categorical() {
            feature_err_arm!(
                self,
                col_ix,
                Categorical,
                u32,
                |row_ix, col_ix| { self.cell(row_ix, col_ix).to_u32_opt() }
            )
        } else {
            panic!("Unsupported feature type");
        };

        Ok(err)
    }

    // Private function impls
    // ---------------------
    fn _logp_unchecked(
        &self,
        col_ixs: &[usize],
        vals: &[Vec<Datum>],
        given: &Given<usize>,
        state_ixs_opt: Option<&[usize]>,
        scaled: bool,
    ) -> Vec<f64> {
        let states = utils::select_states(self.states(), state_ixs_opt);
        let weights = utils::state_weights(&states, col_ixs, given);
        let mut vals_iter = vals.iter();

        let calculator = if scaled {
            utils::Calculator::new_scaled(
                &mut vals_iter,
                &states,
                Some(self.codebook()),
                &weights,
                col_ixs,
            )
        } else {
            utils::Calculator::new(
                &mut vals_iter,
                &states,
                Some(self.codebook()),
                &weights,
                col_ixs,
            )
        };

        calculator.collect()
    }

    fn _simulate_unchecked<R: Rng>(
        &self,
        col_ixs: &[usize],
        given: &Given<usize>,
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

        simulator
            .take(n)
            .map(|row| utils::post_process_row(row, col_ixs, self.codebook()))
            .collect()
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

        let x = utils::pre_process_datum(x.clone(), col_ix, self.codebook())
            .unwrap();

        let state_ixs = state_ixs_opt
            .unwrap_or_else(|| (0..self.n_states()).collect::<Vec<_>>());

        let n_states = state_ixs.len();

        let logp = state_ixs
            .iter()
            .map(|&ix| {
                let state = &self.states()[ix];
                let view_ix = state.asgn().asgn[col_ix];
                let k = state.views[view_ix].asgn().asgn[row_ix];
                state.views[view_ix].ftrs[&col_ix].cpnt_logp(&x, k)
            })
            .logsumexp();
        let s = -logp + (n_states as f64).ln();
        Some(s)
    }

    /// specialization for column pairs. If a specialization is not founds for
    /// the specific columns types, will fall back to MC approximation
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
    /// # Notes
    /// This thing is shit. Don't use it.
    fn _sobol_joint_entropy(&self, col_ixs: &[usize], n: usize) -> f64 {
        let (vals, q_recip) =
            utils::gen_sobol_samples(col_ixs, &self.states()[0], n);
        let logps = self._logp_unchecked(
            col_ixs,
            &vals,
            &Given::<usize>::Nothing,
            None,
            false,
        );
        let h: f64 = logps.iter().map(|logp| -logp * logp.exp()).sum();
        h * q_recip / (n as f64)
    }

    // Use Monte Carlo to estimate the joint entropy
    fn _mc_joint_entropy<R: Rng>(
        &self,
        col_ixs: &[usize],
        n: usize,
        mut rng: &mut R,
    ) -> f64 {
        let states: Vec<_> = self.states().iter().collect();
        let weights =
            utils::given_weights(&states, col_ixs, &Given::<usize>::Nothing);
        // Draws from p(x_1, x_2, ...)
        let mut simulator =
            utils::Simulator::new(&states, &weights, None, col_ixs, &mut rng);
        // Computes ln p (x_1, x_2, ...)
        let calculator = utils::Calculator::new(
            &mut simulator,
            &states,
            None,
            &weights,
            col_ixs,
        );

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
    fn _impute_uncertainty(&self, row_ix: usize, col_ix: usize) -> f64 {
        utils::impute_uncertainty(self.states(), row_ix, col_ix)
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
    ///   designating other observations on which to condition the prediction
    fn _predict_uncertainty(
        &self,
        col_ix: usize,
        given: &Given<usize>,
        state_ixs_opt: Option<&[usize]>,
    ) -> f64 {
        utils::predict_uncertainty(self.states(), col_ix, given, state_ixs_opt)
    }
}

impl<T: CanOracle> OracleT for T {}

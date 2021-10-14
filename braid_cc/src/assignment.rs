//! Data structures for assignments of items to components (partitions)
use braid_stats::mh::mh_prior;
use braid_stats::prior::crp::CrpPrior;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::traits::Rv;
use serde::{Deserialize, Serialize};
use special::Gamma as _;
use thiserror::Error;

use crate::misc::crp_draw;

/// Validates assignments if the `BRAID_NOCHECK` is not set to `"1"`.
macro_rules! validate_assignment {
    ($asgn:expr) => {{
        let validate_asgn: bool = match option_env!("BRAID_NOCHECK") {
            Some(value) => value != "1",
            None => true,
        };
        if validate_asgn {
            $asgn.validate().is_valid()
        } else {
            true
        }
    }};
}

/// Data structure for a data partition and its `Crp` prior
#[allow(dead_code)]
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Assignment {
    /// The `Crp` discount parameter
    pub alpha: f64,
    /// The assignment vector. `asgn[i]` is the partition index of the
    /// i<sup>th</sup> datum.
    pub asgn: Vec<usize>,
    /// Contains the number a data assigned to each partition
    pub counts: Vec<usize>,
    /// The number of partitions/categories
    pub n_cats: usize,
    /// The prior on `alpha`
    pub prior: CrpPrior,
}

/// The possible ways an assignment can go wrong with incorrect bookkeeping
#[derive(Serialize, Deserialize, Eq, PartialEq, Debug, Clone)]
pub struct AssignmentDiagnostics {
    /// There should be a partition with index zero in the assignment vector
    asgn_min_is_zero: bool,
    /// If `n_cats` is `k`, then the largest index in `asgn` should be `k-1`
    asgn_max_is_n_cats_minus_one: bool,
    /// If `n_cats` is `k`, then there should be indices 0, ..., k-1 in the
    /// assignment vector
    asgn_contains_0_through_n_cats_minus_1: bool,
    /// None of the entries in `counts` should be 0
    no_zero_counts: bool,
    /// `counts` should have an entry for every partition/category
    n_cats_cmp_counts_len: bool,
    /// The sum of `counts` should be the number of data
    sum_counts_cmp_n: bool,
    /// The sum of the indices in the assignment vector matches those in
    /// `counts`.
    asgn_agrees_with_counts: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum AssignmentError {
    #[error("Minimum assignment index is not 0")]
    MinAssignmentIndexNotZero,
    #[error("Max assignment index is not n_cats - 1")]
    MaxAssignmentIndexNotNCatsMinusOne,
    #[error("The assignment is missing one or more indices")]
    AssignmentDoesNotContainAllIndices,
    #[error("One or more of the counts is zero")]
    ZeroCounts,
    #[error("The sum of counts does not equal the number of data")]
    SumCountsNotEqualToAssignmentLength,
    #[error("The counts do not agree with the assignment")]
    AssignmentAndCountsDisagree,
    #[error(
        "The length of the counts does not equal the number of categories"
    )]
    NCatsIsNotCountsLength,
    #[error("Attempting to set assignment with a different-length assignment")]
    NewAssignmentLengthMismatch,
}

impl AssignmentDiagnostics {
    pub fn new(asgn: &Assignment) -> Self {
        AssignmentDiagnostics {
            asgn_min_is_zero: { *asgn.asgn.iter().min().unwrap_or(&0) == 0 },
            asgn_max_is_n_cats_minus_one: {
                asgn.asgn
                    .iter()
                    .max()
                    .map(|&x| x == asgn.n_cats - 1)
                    .unwrap_or(true)
            },
            asgn_contains_0_through_n_cats_minus_1: {
                let mut so_far = true;
                for k in 0..asgn.n_cats {
                    so_far = so_far && asgn.asgn.iter().any(|&x| x == k)
                }
                so_far
            },
            no_zero_counts: { !asgn.counts.iter().any(|&ct| ct == 0) },
            n_cats_cmp_counts_len: { asgn.n_cats == asgn.counts.len() },
            sum_counts_cmp_n: {
                let n: usize = asgn.counts.iter().sum();
                n == asgn.asgn.len()
            },
            asgn_agrees_with_counts: {
                let mut all = true;
                for (k, &count) in asgn.counts.iter().enumerate() {
                    let k_count = asgn.asgn.iter().fold(0, |acc, &z| {
                        if z == k {
                            acc + 1
                        } else {
                            acc
                        }
                    });
                    all = all && (k_count == count)
                }
                all
            },
        }
    }

    /// `true` if none of diagnostics was violated
    pub fn is_valid(&self) -> bool {
        self.asgn_min_is_zero
            && self.asgn_max_is_n_cats_minus_one
            && self.asgn_contains_0_through_n_cats_minus_1
            && self.no_zero_counts
            && self.n_cats_cmp_counts_len
            && self.sum_counts_cmp_n
            && self.asgn_agrees_with_counts
    }

    pub fn asgn_min_is_zero(&self) -> Result<(), AssignmentError> {
        if self.asgn_min_is_zero {
            Ok(())
        } else {
            Err(AssignmentError::MinAssignmentIndexNotZero)
        }
    }

    fn asgn_max_is_n_cats_minus_one(&self) -> Result<(), AssignmentError> {
        if self.asgn_max_is_n_cats_minus_one {
            Ok(())
        } else {
            Err(AssignmentError::MaxAssignmentIndexNotNCatsMinusOne)
        }
    }

    fn asgn_contains_0_through_n_cats_minus_1(
        &self,
    ) -> Result<(), AssignmentError> {
        if self.asgn_contains_0_through_n_cats_minus_1 {
            Ok(())
        } else {
            Err(AssignmentError::AssignmentDoesNotContainAllIndices)
        }
    }

    fn no_zero_counts(&self) -> Result<(), AssignmentError> {
        if self.no_zero_counts {
            Ok(())
        } else {
            Err(AssignmentError::ZeroCounts)
        }
    }

    fn n_cats_cmp_counts_len(&self) -> Result<(), AssignmentError> {
        if self.n_cats_cmp_counts_len {
            Ok(())
        } else {
            Err(AssignmentError::NCatsIsNotCountsLength)
        }
    }

    fn sum_counts_cmp_n(&self) -> Result<(), AssignmentError> {
        if self.sum_counts_cmp_n {
            Ok(())
        } else {
            Err(AssignmentError::SumCountsNotEqualToAssignmentLength)
        }
    }

    fn asgn_agrees_with_counts(&self) -> Result<(), AssignmentError> {
        if self.asgn_agrees_with_counts {
            Ok(())
        } else {
            Err(AssignmentError::AssignmentAndCountsDisagree)
        }
    }

    pub fn emit_error(&self) -> Result<(), AssignmentError> {
        let mut results = vec![
            self.asgn_min_is_zero(),
            self.asgn_max_is_n_cats_minus_one(),
            self.asgn_contains_0_through_n_cats_minus_1(),
            self.no_zero_counts(),
            self.n_cats_cmp_counts_len(),
            self.sum_counts_cmp_n(),
            self.asgn_agrees_with_counts(),
        ];
        results.drain(..).collect()
    }
}

/// Constructs `Assignment`s
#[derive(Clone, Debug)]
pub struct AssignmentBuilder {
    n: usize,
    asgn: Option<Vec<usize>>,
    alpha: Option<f64>,
    prior: Option<CrpPrior>,
    seed: Option<u64>,
}

#[derive(Debug, Error, PartialEq)]
pub enum BuildAssignmentError {
    #[error("alpha is zero")]
    AlphaIsZero,
    #[error("non-finite alpha: {alpha}")]
    AlphaNotFinite { alpha: f64 },
    #[error("assignment vector is empty")]
    EmptyAssignmentVec,
    #[error("there are {n_cats} categories but {n} data")]
    NLessThanNCats { n: usize, n_cats: usize },
    #[error("invalid assignment: {0}")]
    AssignmentError(#[from] AssignmentError),
}

impl AssignmentBuilder {
    /// Create a builder for `n`-length assignments
    ///
    /// # Arguments
    /// - n: the number of data/entries in the assignment
    pub fn new(n: usize) -> Self {
        AssignmentBuilder {
            n,
            asgn: None,
            prior: None,
            alpha: None,
            seed: None,
        }
    }

    /// Initialize the builder from an assignment vector
    ///
    /// # Note:
    /// The validity of `asgn` will not be verified until `build` is called.
    pub fn from_vec(asgn: Vec<usize>) -> Self {
        AssignmentBuilder {
            n: asgn.len(),
            asgn: Some(asgn),
            prior: None,
            alpha: None,
            seed: None,
        }
    }

    /// Add a prior on the `Crp` `alpha` parameter
    pub fn with_prior(mut self, prior: CrpPrior) -> Self {
        self.prior = Some(prior);
        self
    }

    /// Use the Geweke `Crp` `alpha` prior
    pub fn with_geweke_prior(mut self) -> Self {
        self.prior = Some(braid_consts::geweke_alpha_prior().into());
        self
    }

    /// Set the `Crp` `alpha` to a specific value
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Set the RNG seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the RNG seed from another RNG
    pub fn seed_from_rng<R: rand::Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// Use a *flat* assignment with one partition
    pub fn flat(mut self) -> Self {
        self.asgn = Some(vec![0; self.n]);
        self
    }

    /// Use an assignment with `n_cats`, evenly populated partitions/categories
    pub fn with_n_cats(
        mut self,
        n_cats: usize,
    ) -> Result<Self, BuildAssignmentError> {
        if n_cats > self.n {
            Err(BuildAssignmentError::NLessThanNCats { n: self.n, n_cats })
        } else {
            let asgn: Vec<usize> = (0..self.n).map(|i| i % n_cats).collect();
            self.asgn = Some(asgn);
            Ok(self)
        }
    }

    /// Build the assignment and consume the builder
    pub fn build(self) -> Result<Assignment, BuildAssignmentError> {
        let prior = self
            .prior
            .unwrap_or_else(|| braid_consts::general_alpha_prior().into());

        let mut rng_opt = if self.alpha.is_none() || self.asgn.is_none() {
            let rng = match self.seed {
                Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
                None => Xoshiro256Plus::from_entropy(),
            };
            Some(rng)
        } else {
            None
        };

        let alpha = match self.alpha {
            Some(alpha) => alpha,
            None => prior.draw(&mut rng_opt.as_mut().unwrap()),
        };

        let n = self.n;
        let asgn = self.asgn.unwrap_or_else(|| {
            crp_draw(n, alpha, &mut rng_opt.as_mut().unwrap()).asgn
        });

        let n_cats: usize = asgn.iter().max().map(|&m| m + 1).unwrap_or(0);
        let mut counts: Vec<usize> = vec![0; n_cats];
        for z in &asgn {
            counts[*z] += 1;
        }

        let asgn_out = Assignment {
            alpha,
            asgn,
            counts,
            n_cats,
            prior,
        };

        if validate_assignment!(asgn_out) {
            Ok(asgn_out)
        } else {
            asgn_out
                .validate()
                .emit_error()
                .map_err(BuildAssignmentError::AssignmentError)
                .map(|_| asgn_out)
        }
    }
}

impl Assignment {
    /// Replace the assignment vector
    pub fn set_asgn(
        &mut self,
        asgn: Vec<usize>,
    ) -> Result<(), AssignmentError> {
        if asgn.len() != self.asgn.len() {
            return Err(AssignmentError::NewAssignmentLengthMismatch);
        }

        let n_cats: usize = *asgn.iter().max().unwrap() + 1;
        let mut counts: Vec<usize> = vec![0; n_cats];
        for z in &asgn {
            counts[*z] += 1;
        }

        self.asgn = asgn;
        self.counts = counts;
        self.n_cats = n_cats;

        if validate_assignment!(self) {
            Ok(())
        } else {
            self.validate().emit_error()
        }
    }

    /// Create and iterator for the assignment vector
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.asgn.iter()
    }

    pub fn len(&self) -> usize {
        self.asgn.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the Dirichlet posterior
    ///
    /// # Arguments
    ///
    /// - append_alpha: if `true` append `alpha` to the end of the vector. This
    ///   is used primarily for the `FiniteCpu` assignment kernel.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_cc::assignment::AssignmentBuilder;
    /// let assignment = AssignmentBuilder::from_vec(vec![0, 0, 1, 2])
    ///     .with_alpha(0.5)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(assignment.asgn, vec![0, 0, 1, 2]);
    /// assert_eq!(assignment.counts, vec![2, 1, 1]);
    /// assert_eq!(assignment.dirvec(false), vec![2.0, 1.0, 1.0]);
    /// assert_eq!(assignment.dirvec(true), vec![2.0, 1.0, 1.0, 0.5]);
    /// ```
    pub fn dirvec(&self, append_alpha: bool) -> Vec<f64> {
        let mut dv: Vec<f64> = self.counts.iter().map(|&x| x as f64).collect();
        if append_alpha {
            dv.push(self.alpha);
        }
        dv
    }

    /// Returns the log of the Dirichlet posterior
    ///
    /// # Arguments
    ///
    /// - append_alpha: if `true` append `alpha` to the end of the vector. This
    ///   is used primarily for the `FiniteCpu` assignment kernel.
    pub fn log_dirvec(&self, append_alpha: bool) -> Vec<f64> {
        let mut dv: Vec<f64> =
            self.counts.iter().map(|&x| (x as f64).ln()).collect();

        if append_alpha {
            dv.push(self.alpha.ln());
        }

        dv
    }

    /// Mark the entry at ix as unassigned. Will remove the entry's contribution
    /// to `n_cats` and `counts`, and will mark `asgn[ix]` with the unassigned
    /// designator..
    pub fn unassign(&mut self, ix: usize) {
        if self.asgn[ix] == usize::max_value() {
            // The row might already be unassigned because it was just
            // inserted and the engine hasn't updated yet.
            return;
        }

        let k = self.asgn[ix];
        if self.counts[k] == 1 {
            self.asgn.iter_mut().for_each(|z| {
                if *z > k {
                    *z -= 1
                }
            });
            let _ct = self.counts.remove(k);
            self.n_cats -= 1;
        } else {
            self.counts[k] -= 1;
        }
        self.asgn[ix] = usize::max_value();
    }

    /// Reassign an unassigned entry
    ///
    /// Returns `Err` if `ix` was not marked as unassigned
    pub fn reassign(&mut self, ix: usize, k: usize) {
        // If the index is the one beyond the number of entries, append k.
        if ix == self.len() {
            self.asgn.push(usize::max_value());
        }

        if self.asgn[ix] != usize::max_value() {
            panic!("Entry {} is assigned. Use assign instead", ix);
        } else if k < self.n_cats {
            self.asgn[ix] = k;
            self.counts[k] += 1;
        } else if k == self.n_cats {
            self.asgn[ix] = k;
            self.n_cats += 1;
            self.counts.push(1);
        } else {
            panic!("k ({}) larger than n_cats ({})", k, self.n_cats);
        }
    }

    /// Append a new, unassigned entry to th end of the assignment
    ///
    /// # Eample
    ///
    /// ```
    /// # use braid_cc::assignment::AssignmentBuilder;
    ///
    /// let mut assignment = AssignmentBuilder::from_vec(vec![0, 0, 1])
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(assignment.asgn, vec![0, 0, 1]);
    ///
    /// assignment.push_unassigned();
    ///
    /// assert_eq!(assignment.asgn, vec![0, 0, 1, usize::max_value()]);
    /// ```
    pub fn push_unassigned(&mut self) {
        self.asgn.push(usize::max_value())
    }

    /// Returns the proportion of data assigned to each partition/category
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_cc::assignment::AssignmentBuilder;
    /// let mut rng = rand::thread_rng();
    /// let assignment = AssignmentBuilder::from_vec(vec![0, 0, 1, 2])
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(assignment.asgn, vec![0, 0, 1, 2]);
    /// assert_eq!(assignment.counts, vec![2, 1, 1]);
    /// assert_eq!(assignment.weights(), vec![0.5, 0.25, 0.25]);
    /// ```
    pub fn weights(&self) -> Vec<f64> {
        let z: f64 = self.len() as f64;
        self.dirvec(false).iter().map(|&w| w / z).collect()
    }

    /// The log of the weights
    pub fn log_weights(&self) -> Vec<f64> {
        self.weights().iter().map(|w| w.ln()).collect()
    }

    /// Posterior update of `alpha` given the prior and the current assignment
    /// vector
    pub fn update_alpha<R: rand::Rng>(
        &mut self,
        n_iter: usize,
        rng: &mut R,
    ) -> f64 {
        // TODO: Should we use a different method to draw CRP alpha that can
        // extend outside of the bulk of the prior's mass?
        let cts = &self.counts;
        let n: usize = self.len();
        let loglike = |alpha: &f64| lcrp(n, cts, *alpha);
        let prior_ref = &self.prior;
        let prior_draw = |rng: &mut R| prior_ref.draw(rng);
        let mh_result =
            mh_prior(self.alpha, loglike, prior_draw, n_iter, rng);
        self.alpha = mh_result.x;
        mh_result.score_x
    }

    /// Validates the assignment
    pub fn validate(&self) -> AssignmentDiagnostics {
        AssignmentDiagnostics::new(self)
    }
}

pub fn lcrp(n: usize, cts: &[usize], alpha: f64) -> f64 {
    let k: f64 = cts.len() as f64;
    let gsum = cts
        .iter()
        .fold(0.0, |acc, ct| acc + (*ct as f64).ln_gamma().0);
    let cpnt_2 = alpha.ln_gamma().0 - (n as f64 + alpha).ln_gamma().0;
    gsum + k.mul_add(alpha.ln(), cpnt_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use rv::dist::Gamma;

    #[test]
    fn zero_count_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![0, 0, 0, 0],
            counts: vec![0, 4],
            n_cats: 1,
            prior: Gamma::new(1.0, 1.0).unwrap().into(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(diagnostic.asgn_max_is_n_cats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_n_cats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.n_cats_cmp_counts_len);
        assert!(!diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn bad_counts_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 3],
            n_cats: 2,
            prior: Gamma::new(1.0, 1.0).unwrap().into(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(diagnostic.asgn_max_is_n_cats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_n_cats_minus_1);
        assert!(!diagnostic.sum_counts_cmp_n);
        assert!(diagnostic.n_cats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn low_n_cats_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 2],
            n_cats: 1,
            prior: Gamma::new(1.0, 1.0).unwrap().into(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_n_cats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_n_cats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.n_cats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn high_n_cats_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 2],
            n_cats: 3,
            prior: Gamma::new(1.0, 1.0).unwrap().into(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_n_cats_minus_one);
        assert!(!diagnostic.asgn_contains_0_through_n_cats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.n_cats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn no_zero_cat_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 2, 2],
            counts: vec![2, 2],
            n_cats: 2,
            prior: Gamma::new(1.0, 1.0).unwrap().into(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(!diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_n_cats_minus_one);
        assert!(!diagnostic.asgn_contains_0_through_n_cats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(diagnostic.n_cats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn drawn_assignment_should_have_valid_partition() {
        let n: usize = 50;

        // do the test 100 times because it's random
        for _ in 0..100 {
            let asgn = AssignmentBuilder::new(n).build().unwrap();
            assert!(asgn.validate().is_valid());
        }
    }

    #[test]
    fn from_prior_should_have_valid_alpha_and_proper_length() {
        let n: usize = 50;
        let asgn = AssignmentBuilder::new(n)
            .with_prior(Gamma::new(1.0, 1.0).unwrap().into())
            .build()
            .unwrap();

        assert!(!asgn.is_empty());
        assert_eq!(asgn.len(), n);
        assert!(asgn.validate().is_valid());
        assert!(asgn.alpha > 0.0);
    }

    #[test]
    fn flat_partition_validation() {
        let n: usize = 50;
        let asgn = AssignmentBuilder::new(n).flat().build().unwrap();

        assert_eq!(asgn.n_cats, 1);
        assert_eq!(asgn.counts.len(), 1);
        assert_eq!(asgn.counts[0], n);
        assert!(asgn.asgn.iter().all(|&z| z == 0));
    }

    #[test]
    fn from_vec() {
        let z = vec![0, 1, 2, 0, 1, 0];
        let asgn = AssignmentBuilder::from_vec(z).build().unwrap();
        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts[0], 3);
        assert_eq!(asgn.counts[1], 2);
        assert_eq!(asgn.counts[2], 1);
    }

    #[test]
    fn with_n_cats_n_cats_evenly_divides_n() {
        let asgn = AssignmentBuilder::new(100)
            .with_n_cats(5)
            .expect("Whoops!")
            .build()
            .unwrap();
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.n_cats, 5);
        assert_eq!(asgn.counts[0], 20);
        assert_eq!(asgn.counts[1], 20);
        assert_eq!(asgn.counts[2], 20);
        assert_eq!(asgn.counts[3], 20);
        assert_eq!(asgn.counts[4], 20);
    }

    #[test]
    fn with_n_cats_n_cats_doesnt_divides_n() {
        let asgn = AssignmentBuilder::new(103)
            .with_n_cats(5)
            .expect("Whoops!")
            .build()
            .unwrap();
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.n_cats, 5);
        assert_eq!(asgn.counts[0], 21);
        assert_eq!(asgn.counts[1], 21);
        assert_eq!(asgn.counts[2], 21);
        assert_eq!(asgn.counts[3], 20);
        assert_eq!(asgn.counts[4], 20);
    }

    #[test]
    fn dirvec_with_alpha_1() {
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.0)
            .build()
            .unwrap();
        let dv = asgn.dirvec(false);

        assert_eq!(dv.len(), 3);
        assert_relative_eq!(dv[0], 3.0, epsilon = 10E-10);
        assert_relative_eq!(dv[1], 2.0, epsilon = 10E-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10E-10);
    }

    #[test]
    fn dirvec_with_alpha_15() {
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.5)
            .build()
            .unwrap();
        let dv = asgn.dirvec(true);

        assert_eq!(dv.len(), 4);
        assert_relative_eq!(dv[0], 3.0, epsilon = 10E-10);
        assert_relative_eq!(dv[1], 2.0, epsilon = 10E-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10E-10);
        assert_relative_eq!(dv[3], 1.5, epsilon = 10E-10);
    }

    #[test]
    fn log_dirvec_with_alpha_1() {
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.0)
            .build()
            .unwrap();
        let ldv = asgn.log_dirvec(false);

        assert_eq!(ldv.len(), 3);
        assert_relative_eq!(ldv[0], 3.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[1], 2.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[2], 1.0_f64.ln(), epsilon = 10E-10);
    }

    #[test]
    fn log_dirvec_with_alpha_15() {
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.5)
            .build()
            .unwrap();

        let ldv = asgn.log_dirvec(true);

        assert_eq!(ldv.len(), 4);
        assert_relative_eq!(ldv[0], 3.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[1], 2.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[2], 1.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[3], 1.5_f64.ln(), epsilon = 10E-10);
    }

    #[test]
    fn weights() {
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.0)
            .build()
            .unwrap();
        let weights = asgn.weights();

        assert_eq!(weights.len(), 3);
        assert_relative_eq!(weights[0], 3.0 / 6.0, epsilon = 10E-10);
        assert_relative_eq!(weights[1], 2.0 / 6.0, epsilon = 10E-10);
        assert_relative_eq!(weights[2], 1.0 / 6.0, epsilon = 10E-10);
    }

    #[test]
    fn lcrp_all_ones() {
        let lcrp_1 = lcrp(4, &vec![1, 1, 1, 1], 1.0);
        assert_relative_eq!(lcrp_1, -3.17805383034795, epsilon = 10E-8);

        let lcrp_2 = lcrp(4, &vec![1, 1, 1, 1], 2.1);
        assert_relative_eq!(lcrp_2, -1.94581759074351, epsilon = 10E-8);
    }

    #[test]
    fn unassign_non_singleton() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut asgn = AssignmentBuilder::from_vec(z).build().unwrap();

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(1);

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 2, 2]);
        assert_eq!(asgn.asgn, vec![0, usize::max_value(), 1, 1, 2, 2]);
    }

    #[test]
    fn unassign_singleton_low() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut asgn = AssignmentBuilder::from_vec(z).build().unwrap();

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(0);

        assert_eq!(asgn.n_cats, 2);
        assert_eq!(asgn.counts, vec![3, 2]);
        assert_eq!(asgn.asgn, vec![usize::max_value(), 0, 0, 0, 1, 1]);
    }

    #[test]
    fn unassign_singleton_high() {
        let z: Vec<usize> = vec![0, 0, 1, 1, 1, 2];
        let mut asgn = AssignmentBuilder::from_vec(z).build().unwrap();

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![2, 3, 1]);

        asgn.unassign(5);

        assert_eq!(asgn.n_cats, 2);
        assert_eq!(asgn.counts, vec![2, 3]);
        assert_eq!(asgn.asgn, vec![0, 0, 1, 1, 1, usize::max_value()]);
    }

    #[test]
    fn unassign_singleton_middle() {
        let z: Vec<usize> = vec![0, 0, 1, 2, 2, 2];
        let mut asgn = AssignmentBuilder::from_vec(z).build().unwrap();

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![2, 1, 3]);

        asgn.unassign(2);

        assert_eq!(asgn.n_cats, 2);
        assert_eq!(asgn.counts, vec![2, 3]);
        assert_eq!(asgn.asgn, vec![0, 0, usize::max_value(), 1, 1, 1]);
    }

    #[test]
    fn reassign_to_existing_cat() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut asgn = AssignmentBuilder::from_vec(z).build().unwrap();

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(1);

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 2, 2]);
        assert_eq!(asgn.asgn, vec![0, usize::max_value(), 1, 1, 2, 2]);

        asgn.reassign(1, 1);

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);
        assert_eq!(asgn.asgn, vec![0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn reassign_to_new_cat() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut asgn = AssignmentBuilder::from_vec(z).build().unwrap();

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(0);

        assert_eq!(asgn.n_cats, 2);
        assert_eq!(asgn.counts, vec![3, 2]);
        assert_eq!(asgn.asgn, vec![usize::max_value(), 0, 0, 0, 1, 1]);

        asgn.reassign(0, 2);

        assert_eq!(asgn.n_cats, 3);
        assert_eq!(asgn.counts, vec![3, 2, 1]);
        assert_eq!(asgn.asgn, vec![2, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn dirvec_with_unassigned_entry() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut asgn = AssignmentBuilder::from_vec(z)
            .with_alpha(1.0)
            .build()
            .unwrap();

        asgn.unassign(5);

        let dv = asgn.dirvec(false);

        assert_eq!(dv.len(), 3);
        assert_relative_eq!(dv[0], 1.0, epsilon = 10e-10);
        assert_relative_eq!(dv[1], 3.0, epsilon = 10e-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10e-10);
    }

    #[test]
    fn manual_seed_control_works() {
        let asgn_1 = AssignmentBuilder::new(25).with_seed(17834795).build();
        let asgn_2 = AssignmentBuilder::new(25).with_seed(17834795).build();
        let asgn_3 = AssignmentBuilder::new(25).build();
        assert_eq!(asgn_1, asgn_2);
        assert_ne!(asgn_1, asgn_3);
    }

    #[test]
    fn from_rng_seed_control_works() {
        let mut rng_1 = Xoshiro256Plus::seed_from_u64(17834795);
        let mut rng_2 = Xoshiro256Plus::seed_from_u64(17834795);
        let asgn_1 =
            AssignmentBuilder::new(25).seed_from_rng(&mut rng_1).build();
        let asgn_2 =
            AssignmentBuilder::new(25).seed_from_rng(&mut rng_2).build();
        let asgn_3 = AssignmentBuilder::new(25).build();
        assert_eq!(asgn_1, asgn_2);
        assert_ne!(asgn_1, asgn_3);
    }
}

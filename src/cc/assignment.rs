extern crate rand;
extern crate rv;
extern crate special;

use self::rand::Rng;
use self::rv::dist::Gamma;
use self::rv::traits::Rv;
use self::special::Gamma as SGamma;
use defaults;
use misc::crp_draw;
use result;
use stats::mh::mh_prior;

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
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Assignment {
    /// The `Crp` discoutn parameter
    pub alpha: f64,
    /// The assignment vector. `asgn[i]` is the partition index of the
    /// i<sup>th</sup> datum.
    pub asgn: Vec<usize>,
    /// Contains the number a data assigned to each partition
    pub counts: Vec<usize>,
    /// The number of partitions/categories
    pub ncats: usize,
    /// The prior on `alpha`
    pub prior: Gamma,
}

/// The possible ways an assignment can go wrong with incorrect bookkeeping
pub struct AssignmentDiagnostics {
    /// There should be a partition with index zero in the assignment vector
    asgn_min_is_zero: bool,
    /// If `ncats` is `k`, then the largest index in `asgn` should be `k-1`
    asgn_max_is_ncats_minus_one: bool,
    /// If `ncats` is `k`, then there should be indices 0, ..., k-1 in the
    /// assignment vector
    asgn_contains_0_through_ncats_minus_1: bool,
    /// None of the entries in `counts` should be 0
    no_zero_counts: bool,
    /// `counts` should have an entry for every partition/category
    ncats_cmp_counts_len: bool,
    /// The sum of `counts` should be the number of data
    sum_counts_cmp_n: bool,
    /// The sum of the indices in the assignment vector matches those in
    /// `counts`.
    asgn_agrees_with_counts: bool,
}

impl AssignmentDiagnostics {
    /// `true` if none of diagnostics was violated
    pub fn is_valid(&self) -> bool {
        self.asgn_min_is_zero
            && self.asgn_max_is_ncats_minus_one
            && self.asgn_contains_0_through_ncats_minus_1
            && self.no_zero_counts
            && self.ncats_cmp_counts_len
            && self.sum_counts_cmp_n
            && self.asgn_agrees_with_counts
    }
}

/// Constructs `Assignment`s
#[derive(Clone, Debug)]
pub struct AssignmentBuilder {
    n: usize,
    asgn: Option<Vec<usize>>,
    alpha: Option<f64>,
    prior: Option<Gamma>,
}

impl AssignmentBuilder {
    /// Create a builder for `n`-length assignments
    ///
    /// # Arguments
    ///
    /// - n: the number of data/entries in the assignment
    pub fn new(n: usize) -> Self {
        AssignmentBuilder {
            n,
            asgn: None,
            prior: None,
            alpha: None,
        }
    }

    /// Initialize the builder from an assignment vector
    ///
    /// # Note:
    ///
    /// The validity of `asgn` will not be verified until `build` is called.
    pub fn from_vec(asgn: Vec<usize>) -> Self {
        AssignmentBuilder {
            n: asgn.len(),
            asgn: Some(asgn),
            prior: None,
            alpha: None,
        }
    }

    /// Add a prior on the `Crp` `alpha` parameter
    pub fn with_prior(mut self, prior: Gamma) -> Self {
        self.prior = Some(prior);
        self
    }

    /// Use the Geweke `Crp` `alpha` prior
    pub fn with_geweke_prior(mut self) -> Self {
        self.prior = Some(defaults::GEWEKE_ALPHA_PRIOR);
        self
    }

    /// Set the `Crp` `alpha` to a specific value
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Use a *flat* assignment with one partition
    pub fn flat(mut self) -> Self {
        self.asgn = Some(vec![0; self.n]);
        self
    }

    /// Use an assignment with `ncats`, evenly populated partitions/categories
    pub fn with_ncats(mut self, ncats: usize) -> result::Result<Self> {
        if ncats > self.n {
            let msg = format!(
                "ncats ({}) exceeds the number of entries ({})",
                ncats, self.n
            );
            let err = result::Error::new(
                result::ErrorKind::DimensionMismatch,
                msg.as_str(),
            );
            Err(err)
        } else {
            let asgn: Vec<usize> = (0..self.n).map(|i| i % ncats).collect();
            self.asgn = Some(asgn);
            Ok(self)
        }
    }

    // TODO: should return Result<assignment>
    /// Build the assignment and consume the builder
    pub fn build<R: Rng>(self, mut rng: &mut R) -> result::Result<Assignment> {
        let prior = self.prior.unwrap_or(defaults::GENERAL_ALPHA_PRIOR);

        let alpha = match self.alpha {
            Some(alpha) => alpha,
            None => prior.draw(&mut rng),
        };

        let asgn = self.asgn.unwrap_or(crp_draw(self.n, alpha, &mut rng).asgn);

        let ncats: usize = *asgn.iter().max().unwrap() + 1;
        let mut counts: Vec<usize> = vec![0; ncats];
        for z in &asgn {
            counts[*z] += 1;
        }

        let asgn_out = Assignment {
            alpha,
            asgn,
            counts,
            ncats,
            prior,
        };

        if validate_assignment!(asgn_out) {
            Ok(asgn_out)
        } else {
            Err(result::Error::new(
                result::ErrorKind::InvalidAssignment,
                "invalid assignment",
            ))
        }
    }
}

impl Assignment {
    /// Replace the assignment vector
    pub fn set_asgn(&mut self, asgn: Vec<usize>) -> result::Result<()> {
        let ncats: usize = *asgn.iter().max().unwrap() + 1;
        let mut counts: Vec<usize> = vec![0; ncats];
        for z in &asgn {
            counts[*z] += 1;
        }

        self.asgn = asgn;
        self.counts = counts;
        self.ncats = ncats;

        if validate_assignment!(self) {
            Ok(())
        } else {
            Err(result::Error::new(
                result::ErrorKind::InvalidAssignment,
                "Provided assignment is invalid",
            ))
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
    /// # extern crate braid;
    /// # extern crate rand;
    /// # use braid::cc::AssignmentBuilder;
    /// let mut rng = rand::thread_rng();
    /// let assignment = AssignmentBuilder::from_vec(vec![0, 0, 1, 2])
    ///     .with_alpha(0.5)
    ///     .build(&mut rng)
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
    /// to `ncats` and `counts`, and will mark `asgn[ix]` with the unassigned
    /// designator..
    pub fn unassign(&mut self, ix: usize) {
        let k = self.asgn[ix];
        if self.counts[k] == 1 {
            self.asgn.iter_mut().for_each(|z| {
                if *z > k {
                    *z -= 1
                }
            });
            let _ct = self.counts.remove(k);
            self.ncats -= 1;
        } else {
            self.counts[k] -= 1;
        }
        self.asgn[ix] = usize::max_value();
    }

    /// Reassign an unassigned entry
    ///
    /// Returns `Err` if `ix` was not marked as unassigned
    pub fn reassign(&mut self, ix: usize, k: usize) -> result::Result<()> {
        // If the index is the one beyond the number of entries, append k.
        if ix == self.len() {
            self.asgn.push(usize::max_value());
        }
        if self.asgn[ix] != usize::max_value() {
            let msg = format!("Entry {} is assigned. Use assign instead", ix);
            Err(result::Error::new(
                result::ErrorKind::AlreadyAssigned,
                msg.as_str(),
            ))
        } else {
            if k < self.ncats {
                self.asgn[ix] = k;
                self.counts[k] += 1;
                Ok(())
            } else if k == self.ncats {
                self.asgn[ix] = k;
                self.ncats += 1;
                self.counts.push(1);
                Ok(())
            } else {
                let msg =
                    format!("k ({}) larger than ncats ({})", k, self.ncats);
                Err(result::Error::new(
                    result::ErrorKind::BoundsError,
                    msg.as_str(),
                ))
            }
        }
    }

    /// Returns the proportion of data assigned to each partition/category
    ///
    /// # Example
    ///
    /// ```rust
    /// # extern crate braid;
    /// # extern crate rand;
    /// # use braid::cc::AssignmentBuilder;
    /// let mut rng = rand::thread_rng();
    /// let assignment = AssignmentBuilder::from_vec(vec![0, 0, 1, 2])
    ///     .build(&mut rng)
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
    pub fn update_alpha<R: Rng>(&mut self, n_iter: usize, mut rng: &mut R) {
        let cts = &self.counts;
        let n: usize = self.len();
        let loglike = |alpha: &f64| lcrp(n, cts, *alpha);
        let prior_ref = &self.prior;
        let prior_draw = |mut rng: &mut R| prior_ref.draw(&mut rng);
        self.alpha =
            mh_prior(self.alpha, loglike, prior_draw, n_iter, &mut rng);
    }

    /// Validates the assignment
    pub fn validate(&self) -> AssignmentDiagnostics {
        AssignmentDiagnostics {
            asgn_min_is_zero: { *self.asgn.iter().min().unwrap() == 0 },
            asgn_max_is_ncats_minus_one: {
                *self.asgn.iter().max().unwrap() == self.ncats - 1
            },
            asgn_contains_0_through_ncats_minus_1: {
                let mut so_far = true;
                for k in 0..self.ncats {
                    so_far = so_far && self.asgn.iter().any(|&x| x == k)
                }
                so_far
            },
            no_zero_counts: { !self.counts.iter().any(|&ct| ct == 0) },
            ncats_cmp_counts_len: { self.ncats == self.counts.len() },
            sum_counts_cmp_n: {
                let n: usize = self.counts.iter().sum();
                n == self.asgn.len()
            },
            asgn_agrees_with_counts: {
                let mut all = true;
                for (k, &count) in self.counts.iter().enumerate() {
                    let k_count = self.asgn.iter().fold(0, |acc, &z| {
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
}

fn lcrp(n: usize, cts: &[usize], alpha: f64) -> f64 {
    let k: f64 = cts.len() as f64;
    let gsum = cts
        .iter()
        .fold(0.0, |acc, ct| acc + (*ct as f64).ln_gamma().0);
    gsum + k * alpha.ln() + alpha.ln_gamma().0 - (n as f64 + alpha).ln_gamma().0
}

#[cfg(test)]
mod tests {
    extern crate serde_test;

    use self::rand::{FromEntropy, XorShiftRng};
    use super::*;

    #[test]
    fn zero_count_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![0, 0, 0, 0],
            counts: vec![0, 4],
            ncats: 1,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(diagnostic.asgn_max_is_ncats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.ncats_cmp_counts_len);
        assert!(!diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn bad_counts_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 3],
            ncats: 2,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(diagnostic.asgn_max_is_ncats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(!diagnostic.sum_counts_cmp_n);
        assert!(diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn low_ncats_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 2],
            ncats: 1,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_ncats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn high_ncats_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 2],
            ncats: 3,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_ncats_minus_one);
        assert!(!diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn no_zero_cat_fails_validation() {
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![1, 1, 2, 2],
            counts: vec![2, 2],
            ncats: 2,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(!diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_ncats_minus_one);
        assert!(!diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn drawn_assignment_should_have_valid_partition() {
        let n: usize = 50;
        let mut rng = XorShiftRng::from_entropy();

        // do the test 100 times because it's random
        for _ in 0..100 {
            let asgn = AssignmentBuilder::new(n).build(&mut rng).unwrap();
            assert!(asgn.validate().is_valid());
        }
    }

    #[test]
    fn from_prior_should_have_valid_alpha_and_proper_length() {
        let n: usize = 50;
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::new(n)
            .with_prior(Gamma::new(1.0, 1.0).unwrap())
            .build(&mut rng)
            .unwrap();

        assert!(!asgn.is_empty());
        assert_eq!(asgn.len(), n);
        assert!(asgn.validate().is_valid());
        assert!(asgn.alpha > 0.0);
    }

    #[test]
    fn flat_partition_validation() {
        let n: usize = 50;
        let mut rng = XorShiftRng::from_entropy();

        let asgn = AssignmentBuilder::new(n).flat().build(&mut rng).unwrap();

        assert_eq!(asgn.ncats, 1);
        assert_eq!(asgn.counts.len(), 1);
        assert_eq!(asgn.counts[0], n);
        assert!(asgn.asgn.iter().all(|&z| z == 0));
    }

    #[test]
    fn from_vec() {
        let z = vec![0, 1, 2, 0, 1, 0];
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::from_vec(z).build(&mut rng).unwrap();
        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts[0], 3);
        assert_eq!(asgn.counts[1], 2);
        assert_eq!(asgn.counts[2], 1);
    }

    #[test]
    fn with_ncats_ncats_evenly_divides_n() {
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::new(100)
            .with_ncats(5)
            .expect("Whoops!")
            .build(&mut rng)
            .unwrap();
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.ncats, 5);
        assert_eq!(asgn.counts[0], 20);
        assert_eq!(asgn.counts[1], 20);
        assert_eq!(asgn.counts[2], 20);
        assert_eq!(asgn.counts[3], 20);
        assert_eq!(asgn.counts[4], 20);
    }

    #[test]
    fn with_ncats_ncats_doesnt_divides_n() {
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::new(103)
            .with_ncats(5)
            .expect("Whoops!")
            .build(&mut rng)
            .unwrap();
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.ncats, 5);
        assert_eq!(asgn.counts[0], 21);
        assert_eq!(asgn.counts[1], 21);
        assert_eq!(asgn.counts[2], 21);
        assert_eq!(asgn.counts[3], 20);
        assert_eq!(asgn.counts[4], 20);
    }

    #[test]
    fn dirvec_with_alpha_1() {
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.0)
            .build(&mut rng)
            .unwrap();
        let dv = asgn.dirvec(false);

        assert_eq!(dv.len(), 3);
        assert_relative_eq!(dv[0], 3.0, epsilon = 10E-10);
        assert_relative_eq!(dv[1], 2.0, epsilon = 10E-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10E-10);
    }

    #[test]
    fn dirvec_with_alpha_15() {
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.5)
            .build(&mut rng)
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
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.0)
            .build(&mut rng)
            .unwrap();
        let ldv = asgn.log_dirvec(false);

        assert_eq!(ldv.len(), 3);
        assert_relative_eq!(ldv[0], 3.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[1], 2.0_f64.ln(), epsilon = 10E-10);
        assert_relative_eq!(ldv[2], 1.0_f64.ln(), epsilon = 10E-10);
    }

    #[test]
    fn log_dirvec_with_alpha_15() {
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.5)
            .build(&mut rng)
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
        let mut rng = XorShiftRng::from_entropy();
        let asgn = AssignmentBuilder::from_vec(vec![0, 1, 2, 0, 1, 0])
            .with_alpha(1.0)
            .build(&mut rng)
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
        let mut rng = XorShiftRng::from_entropy();
        let mut asgn = AssignmentBuilder::from_vec(z).build(&mut rng).unwrap();

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(1);

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 2, 2]);
        assert_eq!(asgn.asgn, vec![0, usize::max_value(), 1, 1, 2, 2]);
    }

    #[test]
    fn unassign_singleton_low() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut rng = XorShiftRng::from_entropy();
        let mut asgn = AssignmentBuilder::from_vec(z).build(&mut rng).unwrap();

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(0);

        assert_eq!(asgn.ncats, 2);
        assert_eq!(asgn.counts, vec![3, 2]);
        assert_eq!(asgn.asgn, vec![usize::max_value(), 0, 0, 0, 1, 1]);
    }

    #[test]
    fn unassign_singleton_high() {
        let z: Vec<usize> = vec![0, 0, 1, 1, 1, 2];
        let mut rng = XorShiftRng::from_entropy();
        let mut asgn = AssignmentBuilder::from_vec(z).build(&mut rng).unwrap();

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![2, 3, 1]);

        asgn.unassign(5);

        assert_eq!(asgn.ncats, 2);
        assert_eq!(asgn.counts, vec![2, 3]);
        assert_eq!(asgn.asgn, vec![0, 0, 1, 1, 1, usize::max_value()]);
    }

    #[test]
    fn reassign_to_existing_cat() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut rng = XorShiftRng::from_entropy();
        let mut asgn = AssignmentBuilder::from_vec(z).build(&mut rng).unwrap();

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(1);

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 2, 2]);
        assert_eq!(asgn.asgn, vec![0, usize::max_value(), 1, 1, 2, 2]);

        asgn.reassign(1, 1).expect("failed to reassign");

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);
        assert_eq!(asgn.asgn, vec![0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn reassign_to_new_cat() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut rng = XorShiftRng::from_entropy();
        let mut asgn = AssignmentBuilder::from_vec(z).build(&mut rng).unwrap();

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![1, 3, 2]);

        asgn.unassign(0);

        assert_eq!(asgn.ncats, 2);
        assert_eq!(asgn.counts, vec![3, 2]);
        assert_eq!(asgn.asgn, vec![usize::max_value(), 0, 0, 0, 1, 1]);

        asgn.reassign(0, 2).expect("failed to reassign");

        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts, vec![3, 2, 1]);
        assert_eq!(asgn.asgn, vec![2, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn dirvec_with_unassigned_entry() {
        let z: Vec<usize> = vec![0, 1, 1, 1, 2, 2];
        let mut rng = XorShiftRng::from_entropy();
        let mut asgn = AssignmentBuilder::from_vec(z)
            .with_alpha(1.0)
            .build(&mut rng)
            .unwrap();

        asgn.unassign(5);

        let dv = asgn.dirvec(false);

        assert_eq!(dv.len(), 3);
        assert_relative_eq!(dv[0], 1.0, epsilon = 10e-10);
        assert_relative_eq!(dv[1], 3.0, epsilon = 10e-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10e-10);
    }
}

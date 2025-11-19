use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use thiserror::Error;

use lace_stats::assignment::{Assignment, AssignmentError};
use lace_stats::prior_process::Process;

/// Constructs `Assignment`s
#[derive(Clone, Debug)]
pub struct AssignmentBuilder {
    n: usize,
    asgn: Option<Vec<usize>>,
    prior_process: Option<Process>,
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
            prior_process: None,
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
            prior_process: None,
            seed: None,
        }
    }

    /// Add a prior on the `Crp` `alpha` parameter
    #[must_use]
    pub fn with_prior_process(mut self, process: Process) -> Self {
        self.prior_process = Some(process);
        self
    }

    /// Set the RNG seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the RNG seed from another RNG
    #[must_use]
    pub fn seed_from_rng<R: rand::Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// Use a *flat* assignment with one partition
    #[must_use]
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
        use lace_stats::prior_process::{Dirichlet, PriorProcessT};

        let mut rng = self.seed.map_or_else(
            || Xoshiro256Plus::from_entropy(),
            Xoshiro256Plus::seed_from_u64,
        );

        let process = self.prior_process.unwrap_or_else(|| {
            Process::Dirichlet(Dirichlet::from_prior(
                lace_consts::general_alpha_prior(),
                &mut rng,
            ))
        });

        let n = self.n;
        let asgn = self
            .asgn
            .unwrap_or_else(|| process.draw_assignment(n, &mut rng).asgn);

        let n_cats: usize = asgn.iter().max().map(|&m| m + 1).unwrap_or(0);
        let mut counts: Vec<usize> = vec![0; n_cats];
        for z in &asgn {
            counts[*z] += 1;
        }

        let asgn_out = Assignment {
            asgn,
            counts,
            n_cats,
        };

        if lace_stats::validate_assignment!(asgn_out) {
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

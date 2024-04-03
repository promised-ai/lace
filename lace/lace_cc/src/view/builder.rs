use super::View;

use crate::feature::ColModel;
use crate::feature::Feature;

use std::collections::BTreeMap;

use lace_stats::assignment::Assignment;
use lace_stats::prior_process::{PriorProcess, PriorProcessT, Process};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

/// Builds a `View`
pub struct Builder {
    n_rows: usize,
    process: Option<Process>,
    asgn: Option<Assignment>,
    ftrs: Option<Vec<ColModel>>,
    seed: Option<u64>,
}

impl Builder {
    /// Start building a view with a given number of rows
    pub fn new(n_rows: usize) -> Self {
        Builder {
            n_rows,
            asgn: None,
            process: None,
            ftrs: None,
            seed: None,
        }
    }

    /// Start building a view with a given row assignment.
    ///
    /// Note that the number of rows will be the assignment length.
    pub fn from_assignment(asgn: Assignment) -> Self {
        Builder {
            n_rows: asgn.len(),
            asgn: Some(asgn),
            process: None, // is ignored in asgn set
            ftrs: None,
            seed: None,
        }
    }

    pub fn from_prior_process(prior_process: PriorProcess) -> Self {
        Builder {
            n_rows: prior_process.asgn.len(),
            asgn: Some(prior_process.asgn),
            process: Some(prior_process.process),
            ftrs: None,
            seed: None,
        }
    }

    /// Put a custom `Gamma` prior on the CRP alpha
    #[must_use]
    pub fn prior_process(mut self, process: Process) -> Self {
        self.process = Some(process);
        self
    }

    /// Add features to the `View`
    #[must_use]
    pub fn features(mut self, ftrs: Vec<ColModel>) -> Self {
        self.ftrs = Some(ftrs);
        self
    }

    /// Set the RNG seed
    #[must_use]
    pub fn seed_from_u64(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the RNG seed from another RNG
    #[must_use]
    pub fn seed_from_rng<R: Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// Build the `View` and consume the builder
    pub fn build(self) -> View {
        use lace_consts::general_alpha_prior;
        use lace_stats::prior_process::Dirichlet;

        let mut rng = match self.seed {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_entropy(),
        };

        let process = self.process.unwrap_or_else(|| {
            Process::Dirichlet(Dirichlet::from_prior(
                general_alpha_prior(),
                &mut rng,
            ))
        });

        let asgn = match self.asgn {
            Some(asgn) => asgn,
            None => process.draw_assignment(self.n_rows, &mut rng),
        };

        let prior_process = PriorProcess { process, asgn };

        let weights = prior_process.weight_vec(false);
        let mut ftr_tree = BTreeMap::new();
        if let Some(mut ftrs) = self.ftrs {
            for mut ftr in ftrs.drain(..) {
                ftr.reassign(&prior_process.asgn, &mut rng);
                ftr_tree.insert(ftr.id(), ftr);
            }
        }

        View {
            ftrs: ftr_tree,
            prior_process,
            weights,
        }
    }
}

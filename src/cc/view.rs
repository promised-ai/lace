extern crate rand;

use std::collections::HashSet;
use std::collections::BTreeMap;
use std::iter::FromIterator;

use self::rand::Rng;
use misc::{massflip, transpose};
use dist::Dirichlet;
use dist::traits::RandomVariate;
use cc::Assignment;
use cc::Feature;


/// View is a multivariate generalization of the standard Diriclet-process
/// mixture model (DPGMM). `View` captures a joint distibution over its
/// columns by assuming the columns are dependent.
pub struct View {
    ftrs: BTreeMap<usize, Box<Feature>>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub alpha: f64,
}


/// The MCMC algorithm to use for row reassignment
#[derive(Clone)]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet appproximation
    FiniteCpu,
    /// OpenCL GPU-parallelized finite Dirichlet appproximation
    FiniteGpu,
    /// Sequential importance samplint split-merge
    SplitMerge,
}


impl View {
    /// Construct a View from a vector of `Box`ed `Feature`s
    pub fn new(mut ftrs: Vec<Box<Feature>>, alpha: f64,
               mut rng: &mut Rng) -> View {
        let nrows = ftrs[0].len();
        let asgn = Assignment::draw(nrows, alpha, &mut rng);
        let weights = asgn.weights();
        for ftr in ftrs.iter_mut() {
            ftr.reassign(&asgn, &mut rng);
        }

        let mut ftrs_tree = BTreeMap::new();
        for ftr in ftrs.drain(0..) {
            ftrs_tree.insert(ftr.id(), ftr);
        }
        View{ftrs: ftrs_tree, asgn: asgn, alpha: alpha, weights: weights}
    }

    // No views
    pub fn empty(n: usize) -> View {
        let alpha = 1.0;
        let asgn = Assignment::flat(n, alpha);
        let ftrs: BTreeMap<usize, Box<Feature>> = BTreeMap::new();
        View{ftrs: ftrs, asgn: asgn, alpha: alpha, weights: vec![1.0]}
    }

    /// Returns the number of rows in the `View`
    pub fn nrows(&self) -> usize {
        self.asgn.asgn.len()
    }

    /// Returns the number of columns in the `View`
    pub fn ncols(&self) -> usize {
        self.ftrs.len()
    }

    /// Update the state of the `View` by running the `View` MCMC transitions
    /// `n_iter` times.
    pub fn update(&mut self, n_iter: usize, alg: RowAssignAlg,
                  mut rng: &mut Rng)
    {
        for _ in 0..n_iter {
            self.reassign(alg.clone(), &mut rng);
        }
    }

    /// Reassign the rows to categories
    pub fn reassign(&mut self, alg: RowAssignAlg, mut rng: &mut Rng) {
        match alg {
            FiniteCpu  => self.reassign_rows_finite_cpu(&mut rng),
            FiniteGpu  => self.reassign_rows_finite_gpu(&mut rng),
            SplitMerge => self.reassign_rows_split_merge(&mut rng),
        }
    }

    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut Rng) {
        let ncats = self.asgn.ncats;
        let nrows = self.nrows();

        self.resample_weights(true, &mut rng);
        self.append_empty_component(&mut rng);

        // initialize log probabilities
        // TODO: This way of initialization may be slow
        let mut logps: Vec<Vec<f64>> = vec![vec![0.0; nrows]; ncats + 1];
        for (k, w) in self.weights.iter().enumerate() {
            logps[k] = vec![w.ln(); nrows];
        }

        for k in 0..(ncats + 1) {
            for (_, ftr) in &self.ftrs {
                ftr.accum_score(&mut logps[k], k);
            }
        }

        let logps_t = transpose(&logps);
        let new_asgn_vec = massflip(logps_t, &mut rng);

        self.integrate_finite_asgn(new_asgn_vec);

        // We resample the weights w/o the CRP alpha appended so that the
        // number of weights matches the number of components
        self.resample_weights(false, &mut rng);
    }

    pub fn resample_weights(&mut self, add_empty_component: bool,
                            mut rng: &mut Rng)
    {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec);
        self.weights = dir.draw(&mut rng)
    }

    pub fn reassign_rows_finite_gpu(&mut self, rng: &mut Rng) {
        unimplemented!();
    }

    pub fn reassign_rows_split_merge(&mut self, rng: &mut Rng) {
        unimplemented!();
    }

    // TODO: when we implement prior param update
    pub fn update_prior_params(&mut self) {
        unimplemented!();
    }

    pub fn update_alpha(&mut self) {
        unimplemented!();
    }

    fn append_empty_component(&mut self, mut rng: &mut Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.append_empty_component(&mut rng);
        }
    }

    fn drop_component(&mut self, k: usize) {
        for ftr in self.ftrs.values_mut() {
            ftr.drop_component(k);
        }
    }

    // Cleanup functions
    fn integrate_finite_asgn(&mut self, mut new_asgn_vec: Vec<usize>) {
        // 1. Find unused components
        let ncats = self.asgn.ncats;
        let all_cats: HashSet<_> = HashSet::from_iter(0..ncats);
        let used_cats = HashSet::from_iter(new_asgn_vec.iter().cloned());
        let mut unused_cats: Vec<&usize> = all_cats.difference(&used_cats)
                                                   .collect();
        unused_cats.sort();
        // needs to be in reverse order, because we want to remove the
        // higher-indexed components first to minimize bookkeeping.
        unused_cats.reverse();

        for &k in unused_cats {
            self.drop_component(k);
            for z in new_asgn_vec.iter_mut() {
                if *z > k { *z -= 1};
            }
        }
        self.asgn = Assignment::from_vec(new_asgn_vec, self.alpha);
        assert!(self.asgn.validate().is_valid());
    }

    /// Insert a new `Feature` into the `View`
    pub fn insert_feature(&mut self, mut ftr: Box<Feature>, mut rng: &mut Rng) {
        let id = ftr.id();
        if self.ftrs.contains_key(&id) {
            panic!("Feature {} already in view", id);
        }
        ftr.reassign(&self.asgn, &mut rng);
        self.ftrs.insert(id, ftr);
    }

    /// Remove and return the `Feature` with `id`. Returns `None` if the `id`
    /// is not found.
    pub fn remove_feature(&mut self, id: usize) -> Option<Box<Feature>> {
        self.ftrs.remove(&id)
    }
}

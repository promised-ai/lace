extern crate rand;


use std::collections::BTreeMap;

use self::rand::Rng;
use misc::{massflip, transpose};
use dist::Dirichlet;
use cc::Assignment;
use cc::Feature;

pub struct View {
    ftrs: BTreeMap<usize, Box<Feature>>,
    asgn: Assignment,
    weights: Vec<f64>,
    alpha: f64,
}


pub enum RowAssignmentAlg {
    FiniteCpu,
    FiniteGpu,
    SplitMerge,
}


impl View {
    // Construtors
    pub fn new(ftrs: BTreeMap<usize, Box<Feature>>, asgn: Assignment,
               weights: Vec<f64>, alpha: f64) -> View
    {
        // TODO: Don't let weights be the wrong size
        View{ftrs: ftrs, asgn: asgn, alpha: alpha, weights: weights}
    }

    // No views
    pub fn empty(n: usize) -> View {
        let alpha = 1.0;
        let asgn = Assignment::flat(n, alpha);
        let ftrs: BTreeMap<usize, Box<Feature>> = BTreeMap::new();
        View{ftrs: ftrs, asgn: asgn, alpha: alpha, weights: vec![1.0]}
    }

    pub fn nrows(&self) -> usize {
        self.asgn.asgn.len()
    }

    pub fn ncols(&self) -> usize {
        self.ftrs.len()
    }

    pub fn reassign_rows(&mut self, alg: RowAssignmentAlg, mut rng: &mut Rng) {
        match alg {
            FiniteCpu  => self.reassign_rows_finite_cpu(&mut rng),
            FiniteGpu  => self.reassign_rows_finite_gpu(&mut rng),
            SplitMerge => self.reassign_rows_split_merge(&mut rng),
        }
    }

    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut Rng) {
        let ncats = self.asgn.ncats;
        // FIXME: Fill in with log weights
        let mut logps: Vec<Vec<f64>> = vec![vec![0.0; self.nrows()]; ncats];
        // FIXME: and the new category
        for k in 0..ncats {
            for (_, ftr) in &self.ftrs {
                ftr.accum_score(&mut logps[k], k);
            }
        }

        let logps_t = transpose(&logps);
        let asgn = massflip(logps_t, &mut rng);
    }

    pub fn reassign_rows_finite_gpu(&mut self, mut rng: &mut Rng) {
        unimplemented!();
    }

    pub fn reassign_rows_split_merge(&mut self, mut rng: &mut Rng) {
        unimplemented!();
    }

    // TODO: when we implement prior param update
    pub fn update_prior_params(&mut self) {
        unimplemented!();
    }

    pub fn update_alpha(&mut self) {
        unimplemented!();
    }


    // Cleanup functions
    pub fn insert_feature(&mut self, ftr: Box<Feature>) {
        unimplemented!();
    }

    pub fn remove_feature(&mut self, ftr_id: usize) {
        unimplemented!();
    }
}

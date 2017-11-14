use cc::Assignment;
use cc::Feature;

pub struct View {
    ftrs: BTreeMap<usize, Box<Feature>>,
    asgn: Assignment,
    alpha: f64,
    alpha_params: (f64, f64),
}


pub enum RowAssignmentAlg {
    finite_approx,
    split_merge,
}


impl View {
    // Construtors
    pub fn new(ftrs: Vec<Box<Feature>>, asgn: Assignment, alpha: f64) -> View {
        unimplemented!();
    }

    // No views
    pub fn empty(n: usize) -> View {
        let alpha = 1.0;
        let asgn = Assignment::flat(n, alpha);
        View(ftrs: BTreeMap<usize, Box<Feature>>::new(), asgn, alpha)
    }

    pub fn reassign_rows(&mut self, alg: RowAssignmentAlg) {
        unimplemented!();
    }

    pub fn update_prior_params(&mut self) {
        unimplemented!();
    }

    pub fn update_alpha(&mut self) {
        unimplemented!();
    }

    // specific sampling algorithms
    pub fn reassign_rows_finite(&mut self) {
        unimplemented!();
    }

    pub fn reassign_rows_split_merge(&mut self) {
        unimplemented!();
    }

    // Cleanup functions
    pub insert_feature(&mut self, ftr: Box<Feature>) {
        unimplemented!();
    }

    pub remove_feature(&mut self, ftr_id: usize) {
        unimplemented!();
    }
}

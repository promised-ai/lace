extern crate serde_yaml;

use std::fs::File;
use std::path::Path;
use std::io::Read;
use std::iter::FromIterator;
use std::collections::HashSet;

use cc::State;


/// Teller answers questions
pub struct Teller {
    /// Vector of data-less states
    pub states: Vec<State>,
}


// TODO: Should this go with ColModel?
pub enum DType {
    Continuous(f64),
    Categorical(u8),
    Binary(bool),
    Missing, // Should carry an error message?
}


impl Teller {
    pub fn new(states: Vec<State>) -> Self {
        Teller{states: states}
    }

    pub fn from_yaml(filenames: Vec<&str>) -> Self {
        // TODO: Input validation
        // TODO: Should return Result<Self>
        let states = filenames.iter().map(|filename| {
            let path = Path::new(&filename);
            let mut file = File::open(&path).unwrap();
            let mut yaml = String::new();
            file.read_to_string(&mut yaml);
            serde_yaml::from_str(&yaml).unwrap()
        }).collect();

        Teller{states: states}
    }

    pub fn nstates(&self) -> usize {
        self.states.len()
    }

    pub fn nrows(&self) -> usize {
        self.states[0].nrows()
    }

    pub fn ncols(&self) -> usize {
        self.states[0].ncols()
    }

    /// Estimated dependence probability between `col_a` and `col_b`
    pub fn depprob(&self, col_a: usize, col_b: usize) -> f64 {
        self.states.iter().fold(0.0, |acc, state| {
            if state.asgn.asgn[col_a] == state.asgn.asgn[col_b] {
                acc + 1.0
            } else {
                acc
            }
        }) / (self.nstates() as f64)
    }

    /// Estimated row similarity between `row_a` and `row_b`
    pub fn rowsim(&self, row_a: usize, row_b: usize,
                  wrt: Option<&Vec<usize>>) -> f64
    {
        self.states.iter().fold(0.0, |acc, state| {
            let view_ixs: Vec<usize> = match wrt {
                Some(col_ixs) => {
                    let asgn = &state.asgn.asgn;
                    let viewset: HashSet<usize> = HashSet::from_iter(
                        col_ixs.iter().map(|&col_ix| asgn[col_ix]));
                    viewset.iter().map(|x| *x).collect()
                },
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
        }) / self.nstates() as f64
    }

    pub fn mutual_information(&self, col_a: usize, col_b: usize) -> f64 {
        unimplemented!();
    }

    /// Joint entropy over columns
    pub fn entropy(&self, ixs: Vec<usize>) -> f64 {
        unimplemented!();
    }

    /// Conditional entropy H(A|B)
    pub fn conditional_entropy(&self, col_a: usize, col_b: usize) -> f64 {
        unimplemented!();
    }

    // TODO: How would these functions look if we used enum instead of float?
    pub fn logp(&self, ixs: Vec<usize>, vals: Vec<f64>,
                given: Option<Vec<(usize, f64)>>) -> f64
    {
        unimplemented!();
    }

    pub fn simulate(&self, ixs: Vec<usize>,
                    given: Option<Vec<(usize, f64)>>) -> Vec<f64>
    {
        unimplemented!();
    }

    pub fn impute(&self, row_ix: usize, col_ix: usize) -> (DType, f64) {
        unimplemented!();
    }

}

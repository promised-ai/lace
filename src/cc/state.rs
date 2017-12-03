extern crate rand;

use self::rand::Rng;

use dist::Dirichlet;
use dist::traits::RandomVariate;
use cc::Feature;
use cc::Assignment;
use cc::view::{View, RowAssignAlg};

// For Geweke
// use self::rand::{Rng, SeedableRng, ChaChaRng};
// use std::collections::BTreeMap;
// use dist::{Gaussian, Dirichlet};
// use dist::prior::NormalInverseGamma;
// use geweke::GewekeReady;
// use cc::DataContainer;
// use cc::Column;


pub struct State<R> where R: Rng {
    pub views: Vec<View>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub alpha: f64,
    pub nrows: usize,
    pub ncols: usize,
    rng: R,
}


/// The MCMC algorithm to use for column reassignment
#[derive(Clone)]
pub enum ColAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    FiniteCpu,
    /// Sequential, enumerative Gibbs
    Gibbs,
}



impl<R> State<R>  where R: Rng {
    pub fn new(mut ftrs: Vec<Box<Feature>>, alpha: f64, mut rng: R) -> Self {
        let ncols = ftrs.len();
        let nrows = ftrs[0].len();
        let asgn = Assignment::draw(ncols, alpha, &mut rng);
        let mut views: Vec<View> = (0..asgn.ncats)
            .map(|_| View::empty(nrows, alpha, &mut rng))
            .collect();

        for (&v, ftr) in asgn.asgn.iter().zip(ftrs.drain(..)) {
            views[v].insert_feature(ftr, &mut rng);
        }
      
        let weights = asgn.weights();

        State{views: views,
              asgn: asgn,
              weights: weights,
              alpha: alpha,
              rng: rng,
              nrows: nrows,
              ncols: ncols}
    }

    // // For Geweke
    // pub fn new_seed(mut ftrs: Vec<Box<Feature>>, alpha: f64,
    //                 seed: Seed) -> Self {
    //     let rng = R::from_seed(seed);
    //     let state: State<ChaChaRng> = State::new(ftrs, alpha, rng);
    //     state
    // }

    pub fn update(&mut self) {
        unimplemented!();
    }

    pub fn reassign(&mut self, alg: ColAssignAlg) {
        match alg {
            ColAssignAlg::FiniteCpu => self.reassign_cols_finite_cpu(),
            ColAssignAlg::Gibbs     => unimplemented!(),
        }
    }

    pub fn reassign_cols_finite_cpu(&mut self) {
        let nviews = self.asgn.ncats;

        self.resample_weights(true);
        self.append_empty_view();

        let mut logps: Vec<Vec<f64>> = Vec::with_capacity(nviews + 1);
        for w in &self.weights {
            logps.push(vec![w.ln(); self.ncols]);
        }

        // for v in 0..(nviews + 1) {
        //     for 
        // }
        unimplemented!();
    }

    pub fn reassign_rows(&mut self, rowAlg: RowAssignAlg) {
        for view in &mut self.views {
            view.reassign(rowAlg.clone(), &mut self.rng);
        }
    }

    pub fn resample_weights(&mut self, add_empty_component: bool) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec);
        self.weights = dir.draw(&mut self.rng)
    }

    fn integrate_finite_asgn(&mut self, mut new_asgn_vec: Vec<usize>) {
        unimplemented!();
    }

    fn append_empty_view(&mut self) {
        let view = View::empty(self.nrows, self.alpha, &mut self.rng);
        self.views.push(view)
    }
}


// Geweke
// ======
// pub struct StateGewekeSettings {
//     /// The number of columns/features in the state
//     pub ncols: usize,
//     /// The number of rows in the state
//     pub nrows: usize,
//     /// The row reassignment algorithm
//     pub row_alg: RowAssignAlg,
//     /// The column reassignment algorithm
//     pub col_alg: ColAssignAlg,
//     // TODO: Add vector of column types
// }


// impl<R> GewekeReady for State<R>
//     where R: Rng + SeedableRng<[u32]>
// {
//     type Output = BTreeMap<String, f64>;
//     type Settings = StateGewekeSettings;

//     // FIXME: need nrows, ncols, and algorithm specification
//     fn from_prior(settings: &StateGewekeSettings, mut rng: &mut Rng) -> Self {
//         // generate Columns
//         let g = Gaussian::new(0.0, 1.0);
//         let mut ftrs: Vec<Box<Feature>> = Vec::with_capacity(settings.ncols);
//         for id in 0..settings.ncols {
//             let data = DataContainer::new(g.sample(settings.nrows, &mut rng));
//             let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0);
//             let column = Box::new(Column::new(id, data, prior));
//             ftrs.push(column);
//         }
//         State::new_geweke(ftrs, 1.0, &mut rng)
//     }

//     fn resample_data(&mut self, _: &StateGewekeSettings, rng: &mut Rng) {
//         for view in &mut self.views {
//             let asgn = &view.asgn;
//             view.resample_data(asgn, rng);
//         }
//     }

//     fn resample_parameters(&mut self, settings: &StateGewekeSettings,
//                            rng: &mut Rng) {
//         match settings.row_alg {
//             RowAssignAlg::FiniteCpu  => self.reassign_rows_finite_cpu(rng),
//             RowAssignAlg::FiniteGpu  => unimplemented!(),
//             RowAssignAlg::SplitMerge => unimplemented!(),
//         }
//     }

//     fn summarize(&self) -> BTreeMap<String, f64> {
//         unimplemented!();
//     }
// }

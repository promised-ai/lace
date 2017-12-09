extern crate rand;

use self::rand::Rng;

use misc::{transpose, massflip, unused_components};
use dist::Dirichlet;
use dist::traits::RandomVariate;
use cc::Feature;
use cc::ColModel;
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


// TODO: does the state need to own the Rng? What happens when we need to use
// a PRng?
impl<R> State<R>  where R: Rng {
    pub fn new(views: Vec<View>, asgn: Assignment, alpha: f64, mut rng: R) -> Self {
        let nrows = views[0].nrows();
        let ncols = asgn.len();
        let weights = asgn.weights();

        State{views: views,
              asgn: asgn,
              weights: weights,
              alpha: alpha,
              rng: rng,
              nrows: nrows,
              ncols: ncols}
    }

    pub fn from_prior(mut ftrs: Vec<ColModel>, alpha: f64,
                      mut rng: R) -> Self
    {
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

    pub fn update(&mut self, n_iter: usize) {
        for _ in 0..n_iter {
            self.reassign(ColAssignAlg::FiniteCpu);
            self.reassign_rows(RowAssignAlg::FiniteCpu);
        }
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

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(self.ncols);
        for (i, &v) in self.asgn.asgn.iter().enumerate() {
            ftrs.push(self.views[v].remove_feature(i).unwrap());
        }

        // TODO: make parallel on features
        for (i, ftr) in ftrs.iter().enumerate() {
            for (v, view) in self.views.iter().enumerate() {
                logps[v][i] += ftr.col_score(&view.asgn);
            }
        }

        let logps_t = transpose(&logps);
        let new_asgn_vec = massflip(logps_t, &mut self.rng);

        self.integrate_finite_asgn(new_asgn_vec, ftrs);
        self.resample_weights(false);
    }

    pub fn reassign_rows(&mut self, row_alg: RowAssignAlg) {
        // TODO: make parallel
        for view in &mut self.views {
            view.reassign(row_alg.clone(), &mut self.rng);
        }
    }

    pub fn resample_weights(&mut self, add_empty_component: bool) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec);
        self.weights = dir.draw(&mut self.rng)
    }

    fn integrate_finite_asgn(&mut self, mut new_asgn_vec: Vec<usize>,
                             mut ftrs: Vec<ColModel>)
    {
        let unused_views = unused_components(self.asgn.ncats + 1, &new_asgn_vec);

        for v in unused_views {
            self.drop_view(v);
            for z in new_asgn_vec.iter_mut() {
                if *z > v { *z -= 1};
            }
        }

        self.asgn = Assignment::from_vec(new_asgn_vec, self.alpha);
        assert!(self.asgn.validate().is_valid());

        for (ftr, &v) in ftrs.drain(..).zip(self.asgn.asgn.iter()) {
            self.views[v].insert_feature(ftr, &mut self.rng)
        }
    }

    fn drop_view(&mut self, v: usize) {
        // view goes out of scope and is dropped
        let _view = self.views.remove(v);
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

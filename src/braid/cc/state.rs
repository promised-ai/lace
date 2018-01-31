extern crate rand;

use self::rand::Rng;

use misc::{transpose, massflip, unused_components};
use dist::Dirichlet;
use dist::traits::RandomVariate;
use cc::Feature;
use cc::ColModel;
use cc::ColModelType;
use cc::Assignment;
use cc::view::{View, RowAssignAlg};


// number of interations used by the MH sampler when updating paramters
const N_MH_ITERS: usize = 50;


#[derive(Serialize, Deserialize, Clone)]
pub struct StateDiagnostics {
    pub loglike: Vec<f64>,
    pub nviews: Vec<usize>,
    pub state_alpha: Vec<f64>,
}

impl Default for StateDiagnostics {
    fn default() -> Self {
        StateDiagnostics{loglike: vec![], nviews: vec![], state_alpha: vec![]}
    }
}


#[derive(Serialize, Deserialize, Clone)]
pub struct State {
    pub views: Vec<View>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub alpha: f64,
    pub loglike: f64,
    pub diagnostics: StateDiagnostics,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}


/// The MCMC algorithm to use for column reassignment
#[derive(Clone)]
pub enum ColAssignAlg {
    /// CPU-parallelized finite Dirichlet approximation
    FiniteCpu,
    /// Sequential, enumerative Gibbs
    Gibbs,
}



impl State {
    pub fn new(views: Vec<View>, asgn: Assignment, alpha: f64) -> Self {
        let weights = asgn.weights();

        let mut state = State{
            views: views,
            asgn: asgn,
            weights: weights,
            alpha: alpha,
            loglike: 0.0,
            diagnostics: StateDiagnostics::default()};
        state.loglike = state.loglike();
        state
    }

    pub fn from_prior(mut ftrs: Vec<ColModel>, _alpha: f64,
                      mut rng: &mut Rng) -> Self {
        let ncols = ftrs.len();
        let nrows = ftrs[0].len();
        let asgn = Assignment::from_prior(ncols, &mut rng);
        let alpha = asgn.alpha;
        let mut views: Vec<View> = (0..asgn.ncats)
            .map(|_| View::empty(nrows, alpha, &mut rng))
            .collect();

        for (&v, ftr) in asgn.asgn.iter().zip(ftrs.drain(..)) {
            views[v].init_feature(ftr, &mut rng);
        }

        let weights = asgn.weights();

        let mut state = State{
            views: views,
            asgn: asgn,
            weights: weights,
            alpha: alpha,
            loglike: 0.0,
            diagnostics: StateDiagnostics::default()};
        state.loglike = state.loglike();
        state
    }

    pub fn get_feature(&self, col_ix: usize) -> &ColModel {
        let view_ix = self.asgn.asgn[col_ix];
        &self.views[view_ix].ftrs[&col_ix]
    }

    pub fn nrows(&self) -> usize {
        self.views[0].nrows()
    }

    pub fn ncols(&self) -> usize {
        self.views.iter().fold(0, |acc, v| acc + v.ncols())
    }

    pub fn update(&mut self, n_iter: usize, mut rng: &mut Rng) {
        for _ in 0..n_iter {
            self.reassign(ColAssignAlg::FiniteCpu, &mut rng);
            self.asgn.update_alpha(N_MH_ITERS, &mut rng);
            self.update_views(RowAssignAlg::FiniteCpu, &mut rng);
            self.push_diagnostics();
        }
    }

    fn push_diagnostics(&mut self) {
        self.diagnostics.loglike.push(self.loglike);
        self.diagnostics.nviews.push(self.asgn.ncats);
        self.diagnostics.state_alpha.push(self.asgn.alpha);
    }

    pub fn reassign(&mut self, alg: ColAssignAlg, mut rng: &mut Rng) {
        match alg {
            ColAssignAlg::FiniteCpu => self.reassign_cols_finite_cpu(&mut rng),
            ColAssignAlg::Gibbs     => unimplemented!(),
        }
    }

    // TODO: collect state likelihood at last iteration
    pub fn reassign_cols_finite_cpu(&mut self, mut rng: &mut Rng) {
        let ncols = self.ncols();
        let nviews = self.asgn.ncats;

        self.resample_weights(true, &mut rng);
        self.append_empty_view(&mut rng);

        let mut logps: Vec<Vec<f64>> = Vec::with_capacity(nviews + 1);
        for w in &self.weights {
            logps.push(vec![w.ln(); ncols]);
        }

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
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
        let new_asgn_vec = massflip(logps_t, &mut rng);
        self.loglike = new_asgn_vec.iter()
            .enumerate()
            .fold(0.0, |acc, (i, z)| acc + logps[*z][i]);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, &mut rng);
        self.resample_weights(false, &mut rng);
    }

    pub fn update_views(&mut self, row_alg: RowAssignAlg, mut rng: &mut Rng) {
        // TODO: make parallel
        for view in &mut self.views {
            view.update(1, row_alg.clone(), &mut rng);
        }
    }

    pub fn loglike(&self) -> f64 {
        let mut loglike: f64 = 0.0;
        for view in &self.views {
            let asgn = &view.asgn;
            for ftr in view.ftrs.values() {
                loglike += ftr.col_score(&asgn);
            }
        }
        loglike
    }

    pub fn resample_weights(&mut self, add_empty_component: bool,
                            mut rng: &mut Rng) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec);
        self.weights = dir.draw(&mut rng)
    }

    fn integrate_finite_asgn(&mut self, mut new_asgn_vec: Vec<usize>,
                             mut ftrs: Vec<ColModel>, mut rng: &mut Rng)
    {
        let unused_views = unused_components(self.asgn.ncats + 1,
                                             &new_asgn_vec);

        for v in unused_views {
            self.drop_view(v);
            for z in new_asgn_vec.iter_mut() {
                if *z > v { *z -= 1};
            }
        }

        self.asgn = Assignment::from_vec(new_asgn_vec, self.alpha);
        assert!(self.asgn.validate().is_valid());

        for (ftr, &v) in ftrs.drain(..).zip(self.asgn.asgn.iter()) {
            self.views[v].insert_feature(ftr, &mut rng)
        }
    }

    fn drop_view(&mut self, v: usize) {
        // view goes out of scope and is dropped
        let _view = self.views.remove(v);
    }

    fn append_empty_view(&mut self, mut rng: &mut Rng) {
        let view = View::empty(self.nrows(), self.alpha, &mut rng);
        self.views.push(view)
    }
}


// Geweke
// ======
use geweke::GewekeResampleData;
use geweke::GewekeModel;
use geweke::GewekeSummarize;
use std::collections::BTreeMap;
use cc::column_model::gen_geweke_col_models;


// FIXME: Only implement for one RNG type to make seeding easier
pub struct StateGewekeSettings {
    /// The number of columns/features in the state
    pub ncols: usize,
    /// The number of rows in the state
    pub nrows: usize,
    /// The row reassignment algorithm
    pub row_alg: RowAssignAlg,
    /// The column reassignment algorithm
    pub col_alg: ColAssignAlg,
    /// Column Model types
    pub cm_types: Vec<ColModelType>,
}


impl StateGewekeSettings {
    pub fn new(nrows: usize, cm_types: Vec<ColModelType>) -> Self {
        StateGewekeSettings {
            ncols: cm_types.len(),
            nrows: nrows,
            row_alg: RowAssignAlg::FiniteCpu,
            col_alg: ColAssignAlg::FiniteCpu,
            cm_types: cm_types,
        }
    }
}


impl GewekeResampleData for State {
    type Settings = StateGewekeSettings;

    fn geweke_resample_data(&mut self, _: Option<&StateGewekeSettings>,
                            mut rng: &mut Rng) {
        for view in &mut self.views {
            view.geweke_resample_data(None, &mut rng);
        }
    }
}


impl GewekeSummarize for State {
    fn geweke_summarize(&self) -> BTreeMap<String, f64> {
        let mut stats = BTreeMap::new();
        stats.insert(String::from("n_views"), self.asgn.ncats as f64);
        for view in &self.views {
            stats.append(&mut view.geweke_summarize());
        }
        stats
    }
}


impl GewekeModel for State {
    fn geweke_from_prior(settings: &StateGewekeSettings, mut rng: &mut Rng)
        -> Self
    {
        // TODO: Generate new rng from randomly-drawn seed
        let ftrs = gen_geweke_col_models(&settings.cm_types, settings.nrows,
                                         &mut rng);
        State::from_prior(ftrs, 1.0, &mut rng)
    }

    fn geweke_step(&mut self, _settings: &StateGewekeSettings,
                   mut rng: &mut Rng)
    {
        self.update(1, &mut rng);
    }
}

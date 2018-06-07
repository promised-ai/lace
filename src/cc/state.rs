extern crate rand;

use std::io;

use self::rand::Rng;
use rayon::prelude::*;

use cc::file_utils::save_state;
use cc::view::View;
use cc::Assignment;
use cc::ColAssignAlg;
use cc::ColModel;
use cc::DType;
use cc::FType;
use cc::Feature;
use cc::FeatureData;
use cc::RowAssignAlg;
use cc::{DEFAULT_COL_ASSIGN_ALG, DEFAULT_ROW_ASSIGN_ALG};
use dist::traits::RandomVariate;
use dist::{Categorical, Dirichlet, Gaussian};
use misc::{massflip, transpose, unused_components};

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
        StateDiagnostics {
            loglike: vec![],
            nviews: vec![],
            state_alpha: vec![],
        }
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

impl State {
    pub fn new(views: Vec<View>, asgn: Assignment, alpha: f64) -> Self {
        let weights = asgn.weights();

        let mut state = State {
            views: views,
            asgn: asgn,
            weights: weights,
            alpha: alpha,
            loglike: 0.0,
            diagnostics: StateDiagnostics::default(),
        };
        state.loglike = state.loglike();
        state
    }

    pub fn from_prior(
        mut ftrs: Vec<ColModel>,
        _alpha: f64,
        mut rng: &mut impl Rng,
    ) -> Self {
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

        let mut state = State {
            views: views,
            asgn: asgn,
            weights: weights,
            alpha: alpha,
            loglike: 0.0,
            diagnostics: StateDiagnostics::default(),
        };
        state.loglike = state.loglike();
        state
    }

    pub fn save(&mut self, dir: &str, id: usize) -> io::Result<()> {
        save_state(dir, self, id)
    }

    pub fn get_feature(&self, col_ix: usize) -> &ColModel {
        let view_ix = self.asgn.asgn[col_ix];
        &self.views[view_ix].ftrs[&col_ix]
    }

    pub fn get_feature_mut(&mut self, col_ix: usize) -> &mut ColModel {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].ftrs.get_mut(&col_ix).unwrap()
    }

    pub fn nrows(&self) -> usize {
        self.views[0].nrows()
    }

    pub fn ncols(&self) -> usize {
        self.views.iter().fold(0, |acc, v| acc + v.ncols())
    }

    pub fn update(
        &mut self,
        n_iter: usize,
        row_asgn_alg: Option<RowAssignAlg>,
        col_asgn_alg: Option<ColAssignAlg>,
        mut rng: &mut impl Rng,
    ) {
        let row_alg = row_asgn_alg.unwrap_or(DEFAULT_ROW_ASSIGN_ALG);
        let col_alg = col_asgn_alg.unwrap_or(DEFAULT_COL_ASSIGN_ALG);
        for _ in 0..n_iter {
            self.reassign(col_alg, &mut rng);
            self.asgn.update_alpha(N_MH_ITERS, &mut rng);
            self.update_views(row_alg, &mut rng);
            self.push_diagnostics();
        }
    }

    fn push_diagnostics(&mut self) {
        self.diagnostics.loglike.push(self.loglike);
        self.diagnostics.nviews.push(self.asgn.ncats);
        self.diagnostics.state_alpha.push(self.asgn.alpha);
    }

    pub fn reassign(&mut self, alg: ColAssignAlg, mut rng: &mut impl Rng) {
        match alg {
            ColAssignAlg::FiniteCpu => self.reassign_cols_finite_cpu(&mut rng),
            ColAssignAlg::Gibbs => unimplemented!(),
        }
    }

    // TODO: collect state likelihood at last iteration
    pub fn reassign_cols_finite_cpu(&mut self, mut rng: &mut impl Rng) {
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
        self.loglike = new_asgn_vec
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, z)| acc + logps[*z][i]);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, &mut rng);
        self.resample_weights(false, &mut rng);
    }

    pub fn update_views(
        &mut self,
        row_alg: RowAssignAlg,
        mut _rng: &mut impl Rng,
    ) {
        // TODO: make parallel
        // for view in &mut self.views {
        //     view.update(1, row_alg.clone(), &mut rng);
        // }
        self.views.par_iter_mut().for_each(|view| {
            let mut thread_rng = rand::thread_rng();
            view.update(1, row_alg.clone(), &mut thread_rng);
        });
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

    pub fn logp_at(&self, row_ix: usize, col_ix: usize) -> Option<f64> {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].logp_at(row_ix, col_ix)
    }

    pub fn get_datum(&self, row_ix: usize, col_ix: usize) -> DType {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].get_datum(row_ix, col_ix).unwrap()
    }

    pub fn resample_weights(
        &mut self,
        add_empty_component: bool,
        mut rng: &mut impl Rng,
    ) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec);
        self.weights = dir.draw(&mut rng)
    }

    fn integrate_finite_asgn(
        &mut self,
        mut new_asgn_vec: Vec<usize>,
        mut ftrs: Vec<ColModel>,
        mut rng: &mut impl Rng,
    ) {
        let unused_views =
            unused_components(self.asgn.ncats + 1, &new_asgn_vec);

        for v in unused_views {
            self.drop_view(v);
            for z in new_asgn_vec.iter_mut() {
                if *z > v {
                    *z -= 1
                };
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

    fn append_empty_view(&mut self, mut rng: &mut impl Rng) {
        let view = View::empty(self.nrows(), self.alpha, &mut rng);
        self.views.push(view)
    }

    pub fn extract_continuous_cpnt(
        &self,
        row_ix: usize,
        col_ix: usize,
    ) -> io::Result<&Gaussian> {
        let view_ix = self.asgn.asgn[col_ix];
        let view = &self.views[view_ix];
        let ftr = &view.ftrs[&col_ix];
        match &ftr {
            ColModel::Continuous(ref f) => {
                let k = view.asgn.asgn[row_ix];
                Ok(&f.components[k])
            }
            _ => {
                let err = io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Could not extract Gaussian",
                );
                Err(err)
            }
        }
    }

    pub fn extract_categorical_cpnt(
        &self,
        row_ix: usize,
        col_ix: usize,
    ) -> io::Result<&Categorical<u8>> {
        let view_ix = self.asgn.asgn[col_ix];
        let view = &self.views[view_ix];
        let ftr = &view.ftrs[&col_ix];
        match &ftr {
            ColModel::Categorical(ref f) => {
                let k = view.asgn.asgn[row_ix];
                Ok(&f.components[k])
            }
            _ => {
                let err = io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Could not extract Categorical",
                );
                Err(err)
            }
        }
    }

    pub fn impute_bounds(&self, col_ix: usize) -> Option<(f64, f64)> {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].ftrs[&col_ix].impute_bounds()
    }

    pub fn take_data(&mut self) -> BTreeMap<usize, FeatureData> {
        let mut data = BTreeMap::new();
        self.views.iter_mut().flat_map(|v| &mut v.ftrs).for_each(
            |(&id, ftr)| {
                data.insert(id, ftr.take_data());
            },
        );
        data
    }

    pub fn drop_data(&mut self) {
        let _data = self.take_data();
    }

    pub fn repop_data(
        &mut self,
        mut data: BTreeMap<usize, FeatureData>,
    ) -> io::Result<()> {
        let err_kind = io::ErrorKind::InvalidData;
        if data.len() != self.ncols() {
            let msg = "Data length and state.ncols differ";
            Err(io::Error::new(err_kind, msg))
        } else if (0..self.ncols()).any(|k| !data.contains_key(&k)) {
            let msg = "Data daes not contain all column IDs";
            Err(io::Error::new(err_kind, msg))
        } else {
            let ids: Vec<usize> = data.keys().map(|id| *id).collect();
            for id in ids {
                let mut data_col = data.remove(&id).unwrap();
                self.get_feature_mut(id).repop_data(data_col)?;
            }
            Ok(())
        }
    }

    // TODO: should this return a DataStore?
    pub fn clone_data(&self) -> BTreeMap<usize, FeatureData> {
        let mut data = BTreeMap::new();
        self.views
            .iter()
            .flat_map(|v| &v.ftrs)
            .for_each(|(&id, ftr)| {
                data.insert(id, ftr.clone_data());
            });
        data
    }
}

// Geweke
// ======
use cc::column_model::gen_geweke_col_models;
use geweke::GewekeModel;
use geweke::GewekeResampleData;
use geweke::GewekeSummarize;
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
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
    pub cm_types: Vec<FType>,
}

// TODO: Add builder
impl StateGewekeSettings {
    pub fn new(nrows: usize, cm_types: Vec<FType>) -> Self {
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

    fn geweke_resample_data(
        &mut self,
        _: Option<&StateGewekeSettings>,
        mut rng: &mut impl Rng,
    ) {
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
    fn geweke_from_prior(
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) -> Self {
        // TODO: Generate new rng from randomly-drawn seed
        let ftrs =
            gen_geweke_col_models(&settings.cm_types, settings.nrows, &mut rng);
        State::from_prior(ftrs, 1.0, &mut rng)
    }

    fn geweke_step(
        &mut self,
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        self.update(
            1,
            Some(settings.row_alg),
            Some(settings.col_alg),
            &mut rng,
        );
    }
}

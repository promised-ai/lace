extern crate indicatif;
extern crate rand;

use std::io;

use self::indicatif::ProgressBar;
use self::rand::Rng;
// use rayon::prelude::*;

use cc::file_utils::save_state;
use cc::transition::StateTransition;
use cc::view::View;
use cc::view::ViewGewekeSettings;
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
use misc::{log_pflip, massflip, transpose, unused_components};

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

    pub fn nviews(&self) -> usize {
        self.views.len()
    }

    pub fn step(
        &mut self,
        row_asgn_alg: RowAssignAlg,
        col_asgn_alg: ColAssignAlg,
        transitions: &Vec<StateTransition>,
        mut rng: &mut impl Rng,
    ) {
        for transition in transitions {
            match transition {
                StateTransition::ColumnAssignment => {
                    self.reassign(col_asgn_alg, &mut rng);
                }
                StateTransition::RowAssignment => {
                    self.reassign_rows(row_asgn_alg, &mut rng);
                }
                StateTransition::StateAlpha => {
                    self.asgn.update_alpha(N_MH_ITERS, &mut rng);
                }
                StateTransition::ViewAlphas => {
                    self.update_view_alphas(&mut rng);
                }
                StateTransition::FeaturePriors => {
                    self.update_feature_priors(&mut rng);
                }
            }
        }
    }

    fn reassign_rows(
        &mut self,
        row_asgn_alg: RowAssignAlg,
        mut rng: &mut impl Rng,
    ) {
        // TODO: parallelize this; use correct seeding
        self.views
            .iter_mut()
            .for_each(|v| v.reassign(row_asgn_alg, &mut rng))
    }

    fn update_view_alphas(&mut self, mut rng: &mut impl Rng) {
        self.views.iter_mut().for_each(|v| v.update_alpha(&mut rng))
    }

    fn update_feature_priors(&mut self, mut rng: &mut impl Rng) {
        self.views
            .iter_mut()
            .for_each(|v| v.update_prior_params(&mut rng))
    }

    pub fn default_transitions() -> Vec<StateTransition> {
        vec![
            StateTransition::ColumnAssignment,
            StateTransition::StateAlpha,
            StateTransition::RowAssignment,
            StateTransition::ViewAlphas,
            StateTransition::FeaturePriors,
        ]
    }

    pub fn update_pb(
        &mut self,
        n_iter: usize,
        row_asgn_alg: Option<RowAssignAlg>,
        col_asgn_alg: Option<ColAssignAlg>,
        transitions: Option<Vec<StateTransition>>,
        mut rng: &mut impl Rng,
        pb: &ProgressBar,
    ) {
        let row_alg = row_asgn_alg.unwrap_or(DEFAULT_ROW_ASSIGN_ALG);
        let col_alg = col_asgn_alg.unwrap_or(DEFAULT_COL_ASSIGN_ALG);
        let ts = transitions.unwrap_or(Self::default_transitions());
        for i in 0..n_iter {
            self.step(row_alg, col_alg, &ts, &mut rng);
            self.push_diagnostics();
            pb.set_message(&format!("item #{}", i + 1));
            pb.inc(1);
        }
        pb.finish_with_message("done");
    }

    pub fn update(
        &mut self,
        n_iter: usize,
        row_asgn_alg: Option<RowAssignAlg>,
        col_asgn_alg: Option<ColAssignAlg>,
        transitions: Option<Vec<StateTransition>>,
        mut rng: &mut impl Rng,
    ) {
        let row_alg = row_asgn_alg.unwrap_or(DEFAULT_ROW_ASSIGN_ALG);
        let col_alg = col_asgn_alg.unwrap_or(DEFAULT_COL_ASSIGN_ALG);
        let ts = transitions.unwrap_or(Self::default_transitions());
        for _ in 0..n_iter {
            self.step(row_alg, col_alg, &ts, &mut rng);
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
            ColAssignAlg::Gibbs => self.reassign_cols_gibbs(&mut rng),
        }
    }

    /// Reassign all columns using the Gibbs transition.
    pub fn reassign_cols_gibbs(&mut self, mut rng: &mut impl Rng) {
        let ncols = self.ncols();

        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut col_ixs: Vec<usize> = (0..ncols).map(|i| i).collect();
        rng.shuffle(&mut col_ixs);

        let mut loglike = 0.0;
        for col_ix in col_ixs {
            let mut ftr = self.extract_ftr(col_ix);
            let mut logps = self.asgn.log_dirvec(true);
            let mut ftr_logps = vec![0.0; logps.len()];

            // might be faster with an iterator?
            for (ix, view) in self.views.iter().enumerate() {
                ftr_logps[ix] += ftr.col_score(&view.asgn);
            }

            logps
                .iter_mut()
                .zip(ftr_logps.iter())
                .for_each(|(lpa, lpf)| *lpa += *lpf);

            // assignment for a hypothetical singleton view
            let nviews = self.nviews();
            let tmp_asgn = Assignment::from_prior(self.nrows(), &mut rng);
            logps[nviews] += ftr.col_score(&tmp_asgn);

            // Gibbs step (draw from categorical)
            let v_new = log_pflip(&logps, &mut rng);
            loglike += logps[v_new];

            self.asgn
                .reassign(col_ix, v_new)
                .expect("Failed to reassign");
            if v_new == nviews {
                let new_view =
                    View::with_assignment(vec![ftr], tmp_asgn, &mut rng);
                self.views.push(new_view);
            } else {
                self.views[v_new].insert_feature(ftr, &mut rng);
            }
            assert!(self.asgn.validate().is_valid());
        }
        self.loglike = loglike;
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
        mut rng: &mut impl Rng,
    ) {
        // TODO: make parallel
        let transitions = View::default_transitions();
        for view in &mut self.views {
            view.update(1, row_alg, &transitions, &mut rng);
        }
        // self.views.par_iter_mut().for_each(|view| {
        //     let mut thread_rng = rand::thread_rng();
        //     view.update(1, row_alg, &mut thread_rng);
        // });
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

    /// Extract a feature from its view, unassign it, and drop the view if it
    /// is a singleton.
    fn extract_ftr(&mut self, ix: usize) -> ColModel {
        let v = self.asgn.asgn[ix];
        let ct = self.asgn.counts[v];
        let ftr = self.views[v].remove_feature(ix).unwrap();
        if ct == 1 {
            self.drop_view(v);
        }
        self.asgn.unassign(ix);
        ftr
    }

    /// Remove the view, but do not adjust any other metadata
    fn drop_view(&mut self, view_ix: usize) {
        // view goes out of scope and is dropped
        let _view = self.views.remove(view_ix);
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
    /// Which transitions to do
    pub transitions: Vec<StateTransition>,
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
            transitions: State::default_transitions(),
        }
    }
}

impl GewekeResampleData for State {
    type Settings = StateGewekeSettings;

    fn geweke_resample_data(
        &mut self,
        settings: Option<&StateGewekeSettings>,
        mut rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        // XXX: View.geweke_resample_data only needs the transitions
        let view_settings = ViewGewekeSettings {
            nrows: 0,
            ncols: 0,
            row_alg: s.row_alg,
            cm_types: vec![],
            transitions: StateTransition::to_view_transitions(&s.transitions),
        };
        for view in &mut self.views {
            view.geweke_resample_data(Some(&view_settings), &mut rng);
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
            Some(settings.transitions.clone()),
            &mut rng,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cc::codebook::ColMetadata;
    use data::StateBuilder;

    #[test]
    fn extract_ftr_non_singleton() {
        let mut rng = rand::thread_rng();
        let mut state = StateBuilder::new()
            .with_rows(50)
            .add_columns(4, ColMetadata::Continuous { hyper: None })
            .with_views(2)
            .build(&mut rng)
            .expect("Failed to build state");

        assert_eq!(state.asgn.asgn, vec![0, 0, 1, 1]);

        let ftr = state.extract_ftr(1);

        assert_eq!(state.nviews(), 2);
        assert_eq!(state.views[0].ftrs.len(), 1);
        assert_eq!(state.views[1].ftrs.len(), 2);

        assert_eq!(state.asgn.asgn, vec![0, usize::max_value(), 1, 1]);
        assert_eq!(state.asgn.counts, vec![1, 2]);
        assert_eq!(state.asgn.ncats, 2);

        assert_eq!(ftr.id(), 1);
    }

    #[test]
    fn extract_ftr_singleton_low() {
        let mut rng = rand::thread_rng();
        let mut state = StateBuilder::new()
            .with_rows(50)
            .add_columns(3, ColMetadata::Continuous { hyper: None })
            .with_views(2)
            .build(&mut rng)
            .expect("Failed to build state");

        assert_eq!(state.asgn.asgn, vec![0, 1, 1]);

        let ftr = state.extract_ftr(0);

        assert_eq!(state.nviews(), 1);
        assert_eq!(state.views[0].ftrs.len(), 2);

        assert_eq!(state.asgn.asgn, vec![usize::max_value(), 0, 0]);
        assert_eq!(state.asgn.counts, vec![2]);
        assert_eq!(state.asgn.ncats, 1);

        assert_eq!(ftr.id(), 0);
    }

    #[test]
    fn gibbs_col_transition_smoke() {
        let mut rng = rand::thread_rng();
        let mut state = StateBuilder::new()
            .with_rows(50)
            .add_columns(10, ColMetadata::Continuous { hyper: None })
            .with_views(4)
            .with_cats(5)
            .build(&mut rng)
            .expect("Failed to build state");
        state.update(100, None, Some(ColAssignAlg::Gibbs), None, &mut rng);
    }

    #[test]
    fn gibbs_row_transition_smoke() {
        let mut rng = rand::thread_rng();
        let mut state = StateBuilder::new()
            .with_rows(10)
            .add_columns(10, ColMetadata::Continuous { hyper: None })
            .with_views(4)
            .with_cats(5)
            .build(&mut rng)
            .expect("Failed to build state");
        state.update(20, Some(RowAssignAlg::Gibbs), None, None, &mut rng);
    }
}

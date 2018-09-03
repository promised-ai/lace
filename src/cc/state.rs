extern crate indicatif;
extern crate rand;
extern crate rv;

use std::f64::NEG_INFINITY;
use std::io;

use self::indicatif::ProgressBar;
use self::rand::{Rng, SeedableRng, XorShiftRng};
use self::rv::dist::{Categorical, Dirichlet, Gaussian, InvGamma};
use self::rv::misc::ln_pflip;
use self::rv::traits::*;
use cc::file_utils::save_state;
use cc::transition::StateTransition;
use cc::view::ViewGewekeSettings;
use cc::view::{View, ViewBuilder};
use cc::ColAssignAlg;
use cc::ColModel;
use cc::DType;
use cc::FType;
use cc::Feature;
use cc::FeatureData;
use cc::RowAssignAlg;
use cc::{Assignment, AssignmentBuilder};
use cc::{DEFAULT_COL_ASSIGN_ALG, DEFAULT_ROW_ASSIGN_ALG};
use misc::{massflip, unused_components};
use rayon::prelude::*;

// number of interations used by the MH sampler when updating paramters
const N_MH_ITERS: usize = 50;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
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
    pub view_alpha_prior: InvGamma,
    pub loglike: f64,
    pub diagnostics: StateDiagnostics,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}

impl State {
    pub fn new(
        views: Vec<View>,
        asgn: Assignment,
        view_alpha_prior: InvGamma,
    ) -> Self {
        let weights = asgn.weights();

        let mut state = State {
            views: views,
            asgn: asgn,
            weights: weights,
            view_alpha_prior: view_alpha_prior,
            loglike: 0.0,
            diagnostics: StateDiagnostics::default(),
        };
        state.loglike = state.loglike();
        state
    }

    pub fn from_prior(
        mut ftrs: Vec<ColModel>,
        state_alpha_prior: InvGamma,
        view_alpha_prior: InvGamma,
        mut rng: &mut impl Rng,
    ) -> Self {
        let ncols = ftrs.len();
        let nrows = ftrs[0].len();
        let asgn = AssignmentBuilder::new(ncols)
            .with_prior(state_alpha_prior)
            .build(&mut rng);

        let mut views: Vec<View> = (0..asgn.ncats)
            .map(|_| {
                ViewBuilder::new(nrows)
                    .with_alpha_prior(view_alpha_prior.clone())
                    .expect("Invalid prior")
                    .build(&mut rng)
            }).collect();

        for (&v, ftr) in asgn.asgn.iter().zip(ftrs.drain(..)) {
            views[v].init_feature(ftr, &mut rng);
        }

        let weights = asgn.weights();

        let mut state = State {
            views: views,
            asgn: asgn,
            weights: weights,
            view_alpha_prior: view_alpha_prior,
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
                StateTransition::ComponentParams => {
                    self.update_component_params(&mut rng);
                }
            }
        }
    }

    fn reassign_rows(
        &mut self,
        row_asgn_alg: RowAssignAlg,
        mut rng: &mut impl Rng,
    ) {
        let mut rngs: Vec<XorShiftRng> = (0..self.nviews())
            .map(|_| XorShiftRng::from_rng(&mut rng).unwrap())
            .collect();

        self.views.par_iter_mut().zip(rngs.par_iter_mut()).for_each(
            |(view, mut vrng)| {
                view.reassign(row_asgn_alg, &mut vrng);
            },
        );
    }

    fn update_view_alphas(&mut self, mut rng: &mut impl Rng) {
        self.views.iter_mut().for_each(|v| v.update_alpha(&mut rng))
    }

    fn update_feature_priors(&mut self, mut rng: &mut impl Rng) {
        self.views
            .iter_mut()
            .for_each(|v| v.update_prior_params(&mut rng))
    }

    fn update_component_params(&mut self, mut rng: &mut impl Rng) {
        self.views
            .iter_mut()
            .for_each(|v| v.update_component_params(&mut rng))
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
        // info!("Reassigning columns");
        match alg {
            ColAssignAlg::FiniteCpu => self.reassign_cols_finite_cpu(&mut rng),
            ColAssignAlg::Gibbs => self.reassign_cols_gibbs(&mut rng),
            ColAssignAlg::Slice => self.reassign_cols_slice(&mut rng),
        }
    }

    /// Insert new features into the `State`
    pub fn insert_new_features(
        &mut self,
        mut ftrs: Vec<ColModel>,
        mut rng: &mut impl Rng,
    ) -> io::Result<()> {
        ftrs.drain(..)
            .map(|mut ftr| {
                if ftr.len() != self.nrows() {
                    let msg = format!(
                        "State has {} rows, but feature has {}",
                        self.nrows(),
                        ftr.len()
                    );
                    let err = io::Error::new(
                        io::ErrorKind::InvalidInput,
                        msg.as_str(),
                    );
                    Err(err)
                } else {
                    ftr.set_id(self.ncols()); // increases as features inserted
                    self.insert_feature(ftr, &mut rng);
                    Ok(())
                }
            }).collect()
    }

    /// Insert an unassigned feature into the `State` via the `Gibbs`
    /// algorithm. If the feature is new, it is appended to the end of the
    /// `State`.
    pub fn insert_feature(
        &mut self,
        ftr: ColModel,
        mut rng: &mut impl Rng,
    ) -> f64 {
        let col_ix = ftr.id();

        // score for each view
        let mut logps = self.asgn.log_dirvec(true);

        // maintain a vec that  holds just the likelihoods
        let mut ftr_logps: Vec<f64> = Vec::with_capacity(logps.len());

        // TODO: might be faster with an iterator?
        for (ix, view) in self.views.iter().enumerate() {
            let lp = ftr.asgn_score(&view.asgn);
            ftr_logps.push(lp);
            logps[ix] += lp;
        }

        let nviews = self.nviews();
        assert_eq!(nviews + 1, logps.len());

        // assignment for a hypothetical singleton view
        // FIXME: How to handle if we've fixed the view alpha, e.g. in the
        // case where we don't want to sample it for Geweke?
        let tmp_asgn = AssignmentBuilder::new(self.nrows())
            .with_prior(self.view_alpha_prior.clone())
            .build(&mut rng);

        // log likelihood of singleton feature
        // TODO: add `m` in {1, 2, ...} parameter that dictates how many
        // singletons to try.
        let singleton_logp = ftr.asgn_score(&tmp_asgn);
        ftr_logps.push(singleton_logp);
        logps[nviews] += singleton_logp;

        // Gibbs step (draw from categorical)
        let v_new = ln_pflip(&logps, 1, false, &mut rng)[0];

        self.asgn
            .reassign(col_ix, v_new)
            .expect("Failed to reassign");

        if v_new == nviews {
            let new_view =
                ViewBuilder::from_assignment(tmp_asgn).build(&mut rng);
            self.views.push(new_view);
        }
        self.views[v_new].insert_feature(ftr, &mut rng);
        ftr_logps[v_new]
    }

    /// Reassign all columns using the Gibbs transition.
    pub fn reassign_cols_gibbs(&mut self, mut rng: &mut impl Rng) {
        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut col_ixs: Vec<usize> = (0..self.ncols()).map(|i| i).collect();
        rng.shuffle(&mut col_ixs);

        let mut loglike = 0.0;
        for col_ix in col_ixs {
            let mut ftr = self.extract_ftr(col_ix);
            loglike += self.insert_feature(ftr, &mut rng);
            assert!(self.asgn.validate().is_valid());
        }
        self.loglike = loglike;
    }

    /// Reassign columns to views using the `FiniteCpu` transition
    pub fn reassign_cols_finite_cpu(&mut self, mut rng: &mut impl Rng) {
        let ncols = self.ncols();

        self.resample_weights(true, &mut rng);
        self.append_empty_view(&mut rng);

        let log_weights: Vec<f64> =
            self.weights.iter().map(|w| w.ln()).collect();
        let ncats = self.asgn.ncats + 1;

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
        for (i, &v) in self.asgn.asgn.iter().enumerate() {
            ftrs.push(self.views[v].remove_feature(i).unwrap());
        }

        let logps: Vec<Vec<f64>> = ftrs
            .par_iter()
            .map(|ftr| {
                self.views
                    .iter()
                    .enumerate()
                    .map(|(v, view)| {
                        ftr.asgn_score(&view.asgn) + log_weights[v]
                    }).collect()
            }).collect();

        let new_asgn_vec = massflip(logps.clone(), &mut rng);

        // TODO: figure out how to compute this from logps so we don't have
        // to clone logps.
        self.loglike = new_asgn_vec
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, z)| acc + logps[i][*z]);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, ncats, &mut rng);
        self.resample_weights(false, &mut rng);
    }

    /// Reassign columns to views using the improved slice sampler
    pub fn reassign_cols_slice(&mut self, mut rng: &mut impl Rng) {
        use dist::stick_breaking::sb_slice_extend;

        let ncols = self.ncols();

        let udist = self::rand::distributions::Open01;

        self.resample_weights(true, &mut rng);
        let us: Vec<f64> = self
            .asgn
            .asgn
            .iter()
            .map(|&zi| {
                let wi: f64 = self.weights[zi];
                let u: f64 = rng.sample(udist);
                u * wi
            }).collect();

        let u_star: f64 =
            us.iter()
                .fold(1.0, |umin, &ui| if ui < umin { ui } else { umin });

        let weights = sb_slice_extend(
            self.weights.clone(),
            self.asgn.alpha,
            u_star,
            &mut rng,
        ).expect("Failed to break sticks in col assignment");

        let n_new_views = weights.len() - self.weights.len() + 1;
        let nviews = weights.len();

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
        for (i, &v) in self.asgn.asgn.iter().enumerate() {
            ftrs.push(self.views[v].remove_feature(i).unwrap());
        }

        for _ in 0..n_new_views {
            self.append_empty_view(&mut rng);
        }

        // initialize truncated log probabilities
        let logps: Vec<Vec<f64>> = ftrs
            .par_iter()
            .zip(us.par_iter())
            .map(|(ftr, ui)| {
                self.views
                    .iter()
                    .zip(weights.iter())
                    .map(|(view, w)| {
                        if w >= ui {
                            ftr.asgn_score(&view.asgn)
                        } else {
                            NEG_INFINITY
                        }
                    }).collect()
            }).collect();

        let new_asgn_vec = massflip(logps.clone(), &mut rng);

        // TODO: figure out how to compute this from logps so we don't have
        // to clone logps.
        self.loglike = new_asgn_vec
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, z)| acc + logps[i][*z]);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, nviews, &mut rng);
        self.resample_weights(false, &mut rng);
    }

    pub fn loglike(&self) -> f64 {
        let mut loglike: f64 = 0.0;
        for view in &self.views {
            let asgn = &view.asgn;
            for ftr in view.ftrs.values() {
                loglike += ftr.asgn_score(&asgn);
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
        // info!("Resampling weights");
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec).unwrap();
        self.weights = dir.draw(&mut rng)
    }

    fn integrate_finite_asgn(
        &mut self,
        mut new_asgn_vec: Vec<usize>,
        mut ftrs: Vec<ColModel>,
        nviews: usize,
        mut rng: &mut impl Rng,
    ) {
        let unused_views = unused_components(nviews, &new_asgn_vec);

        for v in unused_views {
            self.drop_view(v);
            for z in new_asgn_vec.iter_mut() {
                if *z > v {
                    *z -= 1
                };
            }
        }

        self.asgn.set_asgn(new_asgn_vec);
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
        let view = ViewBuilder::new(self.nrows()).build(&mut rng);
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
                Ok(&f.components[k].fx)
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
    ) -> io::Result<&Categorical> {
        let view_ix = self.asgn.asgn[col_ix];
        let view = &self.views[view_ix];
        let ftr = &view.ftrs[&col_ix];
        match &ftr {
            ColModel::Categorical(ref f) => {
                let k = view.asgn.asgn[row_ix];
                Ok(&f.components[k].fx)
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

    // Forget and re-observe all the data
    // since the data change during the gewek posterior chain runs, the
    // suffstats get out of wack, so we need to re-obseve the new data.
    fn refresh_suffstats(&mut self, mut rng: &mut impl Rng) {
        self.views
            .iter_mut()
            .for_each(|v| v.refresh_suffstats(&mut rng));
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
    fn geweke_summarize(
        &self,
        settings: &StateGewekeSettings,
    ) -> BTreeMap<String, f64> {
        let mut stats = BTreeMap::new();

        let do_col_asgn_transition = settings
            .transitions
            .iter()
            .any(|&t| t == StateTransition::ColumnAssignment);

        let do_alpha_transition = settings
            .transitions
            .iter()
            .any(|&t| t == StateTransition::StateAlpha);

        if do_col_asgn_transition {
            stats.insert(String::from("n_views"), self.asgn.ncats as f64);
        };

        if do_alpha_transition {
            stats.insert(String::from("state CRP alpha"), self.asgn.alpha);
        }

        // Dummy settings. the only thing the view summarizer cares about is the
        // transitions.
        let settings = ViewGewekeSettings {
            ncols: 0,
            nrows: 0,
            row_alg: settings.row_alg,
            cm_types: vec![],
            transitions: StateTransition::to_view_transitions(
                &settings.transitions,
            ),
        };
        for view in &self.views {
            // TODO call out ncats and CRP alpha and construct consolodated
            // stats (e.g. mean or sum)
            stats.append(&mut view.geweke_summarize(&settings));
        }
        stats
    }
}

// XXX: Note that the only geweke is only guaranteed to return turn results if
// all transitions are on. For example, we can turn off the view alphas
// transition, but the Gibbs column transition will create new views with
// alpha drawn from the prior. As of now, the State has no way of knowing that
// the View alphas are 'turned off', so it initializes new Views from the
// prior. So yeah, make sure that all transitions work, and maybe later we'll
// build knowledge of the transition set into the state.
impl GewekeModel for State {
    fn geweke_from_prior(
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) -> Self {
        let has_transition = |t: StateTransition, s: &StateGewekeSettings| {
            s.transitions.iter().any(|&ti| ti == t)
        };
        // TODO: Generate new rng from randomly-drawn seed
        // TODO: Draw features properly depending on the transitions
        let do_ftr_prior_transition =
            has_transition(StateTransition::FeaturePriors, &settings);

        let do_state_alpha_transition =
            has_transition(StateTransition::StateAlpha, &settings);

        let do_view_alphas_transition =
            has_transition(StateTransition::ViewAlphas, &settings);

        let do_col_asgn_transition =
            has_transition(StateTransition::ColumnAssignment, &settings);

        let do_row_asgn_transition =
            has_transition(StateTransition::RowAssignment, &settings);

        let mut ftrs = gen_geweke_col_models(
            &settings.cm_types,
            settings.nrows,
            do_ftr_prior_transition,
            &mut rng,
        );

        let ncols = ftrs.len();

        let asgn_bldr = if do_col_asgn_transition {
            AssignmentBuilder::new(ncols)
        } else {
            AssignmentBuilder::new(ncols).flat()
        }.with_geweke_prior();

        let asgn = if do_state_alpha_transition {
            asgn_bldr.build(&mut rng)
        } else {
            asgn_bldr.with_alpha(1.0).build(&mut rng)
        };

        let view_asgn_bldr = if do_row_asgn_transition {
            if do_view_alphas_transition {
                AssignmentBuilder::new(settings.nrows)
            } else {
                AssignmentBuilder::new(settings.nrows).with_alpha(1.0)
            }
        } else {
            if do_view_alphas_transition {
                AssignmentBuilder::new(settings.nrows).flat()
            } else {
                AssignmentBuilder::new(settings.nrows)
                    .flat()
                    .with_alpha(1.0)
            }
        }.with_geweke_prior();

        let mut views: Vec<View> = (0..asgn.ncats)
            .map(|_| {
                let asgn = view_asgn_bldr.clone().build(&mut rng);
                ViewBuilder::from_assignment(asgn).build(&mut rng)
            }).collect();

        for (&v, ftr) in asgn.asgn.iter().zip(ftrs.drain(..)) {
            views[v].init_feature(ftr, &mut rng);
        }

        let view_alpha_prior = views[0].asgn.prior.clone();

        let weights = asgn.weights();
        State {
            views: views,
            asgn: asgn,
            weights: weights,
            view_alpha_prior: view_alpha_prior,
            loglike: 0.0,
            diagnostics: StateDiagnostics::default(),
        }
    }

    fn geweke_step(
        &mut self,
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        self.refresh_suffstats(&mut rng);
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

    struct StateFlatnessResult {
        pub rows_always_flat: bool,
        pub cols_always_flat: bool,
        pub state_alpha_1: bool,
        pub view_alphas_1: bool,
    }

    fn test_asgn_flatness(
        settings: &StateGewekeSettings,
        n_runs: usize,
        mut rng: &mut impl Rng,
    ) -> StateFlatnessResult {
        let mut cols_always_flat = true;
        let mut rows_always_flat = true;
        let mut state_alpha_1 = true;
        let mut view_alphas_1 = true;

        let basically_one = |x: f64| (x - 1.0).abs() < 1E-12;

        for _ in 0..n_runs {
            let state = State::geweke_from_prior(&settings, &mut rng);
            // 1. Check the assignment prior
            assert_relative_eq!(state.asgn.prior.scale, 3.0, epsilon = 1E-12);
            assert_relative_eq!(state.asgn.prior.shape, 3.0, epsilon = 1E-12);
            // Column assignment is not flat
            if state.asgn.asgn.iter().any(|&zi| zi != 0) {
                cols_always_flat = false;
            }

            if !basically_one(state.asgn.alpha) {
                state_alpha_1 = false;
            }
            // 2. Check the column priors
            for view in state.views.iter() {
                // Check the view assignment priors
                assert_relative_eq!(
                    view.asgn.prior.scale,
                    3.0,
                    epsilon = 1E-12
                );
                assert_relative_eq!(
                    view.asgn.prior.shape,
                    3.0,
                    epsilon = 1E-12
                );
                // Check the view assignments aren't flat
                if view.asgn.asgn.iter().any(|&zi| zi != 0) {
                    rows_always_flat = false;
                }

                if !basically_one(view.asgn.alpha) {
                    view_alphas_1 = false;
                }
            }
        }

        StateFlatnessResult {
            rows_always_flat,
            cols_always_flat,
            state_alpha_1,
            view_alphas_1,
        }
    }

    #[test]
    fn geweke_from_prior_all_transitions() {
        let settings =
            StateGewekeSettings::new(50, vec![FType::Continuous; 40]);
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 10, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_row_transition() {
        let settings = StateGewekeSettings {
            ncols: 20,
            nrows: 50,
            row_alg: RowAssignAlg::FiniteCpu,
            col_alg: ColAssignAlg::FiniteCpu,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment,
                StateTransition::StateAlpha,
                StateTransition::ViewAlphas,
                StateTransition::FeaturePriors,
            ],
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_col_transition() {
        let settings = StateGewekeSettings {
            ncols: 20,
            nrows: 50,
            row_alg: RowAssignAlg::FiniteCpu,
            col_alg: ColAssignAlg::FiniteCpu,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::RowAssignment,
                StateTransition::StateAlpha,
                StateTransition::ViewAlphas,
                StateTransition::FeaturePriors,
            ],
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_row_or_col_transition() {
        let settings = StateGewekeSettings {
            ncols: 20,
            nrows: 50,
            row_alg: RowAssignAlg::FiniteCpu,
            col_alg: ColAssignAlg::FiniteCpu,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::StateAlpha,
                StateTransition::ViewAlphas,
                StateTransition::FeaturePriors,
            ],
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(result.rows_always_flat);
        assert!(result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_alpha_transition() {
        let settings = StateGewekeSettings {
            ncols: 20,
            nrows: 50,
            row_alg: RowAssignAlg::FiniteCpu,
            col_alg: ColAssignAlg::FiniteCpu,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment,
                StateTransition::RowAssignment,
                StateTransition::FeaturePriors,
            ],
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(result.state_alpha_1);
        assert!(result.view_alphas_1);
    }

}

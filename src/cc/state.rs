use std::convert::TryInto;
use std::f64::NEG_INFINITY;
use std::io;
use std::path::Path;
use std::time::Instant;

use braid_flippers::massflip_slice_mat_par;
use braid_stats::prior::crp::CrpPrior;
use braid_stats::{Datum, MixtureType};
use braid_utils::{unused_components, Matrix};
use rand::seq::SliceRandom as _;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use rv::dist::Dirichlet;
use rv::misc::ln_pflip;
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::cc::config::{StateOutputInfo, StateUpdateConfig};
use crate::cc::feature::Component;
use crate::cc::file_utils::{path_validator, save_state};
use crate::cc::view::{
    GewekeViewSummary, View, ViewBuilder, ViewGewekeSettings,
};
use crate::cc::{
    Assignment, AssignmentBuilder, ColAssignAlg, ColModel, FType, Feature,
    FeatureData, RowAssignAlg, StateTransition,
};
use crate::file_config::FileConfig;
use crate::misc::massflip;

/// Stores some diagnostic info in the `State` at every iteration
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct StateDiagnostics {
    /// Log likelihood
    #[serde(default)]
    pub loglike: Vec<f64>,
    /// Log prior likelihood
    #[serde(default)]
    pub log_prior: Vec<f64>,
    /// The number of views
    #[serde(default)]
    pub nviews: Vec<usize>,
    /// The state CRP alpha
    #[serde(default)]
    pub state_alpha: Vec<f64>,
    /// The number of categories in the views with the fewest categories
    #[serde(default)]
    pub ncats_min: Vec<usize>,
    /// The number of categories in the views with the most categories
    #[serde(default)]
    pub ncats_max: Vec<usize>,
    /// The median number of categories in a view
    #[serde(default)]
    pub ncats_median: Vec<f64>,
}

impl Default for StateDiagnostics {
    fn default() -> Self {
        StateDiagnostics {
            loglike: vec![],
            log_prior: vec![],
            nviews: vec![],
            state_alpha: vec![],
            ncats_min: vec![],
            ncats_max: vec![],
            ncats_median: vec![],
        }
    }
}

/// A cross-categorization state
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct State {
    /// The views of columns
    pub views: Vec<View>,
    /// The assignment of columns to views
    pub asgn: Assignment,
    /// The weights of each view in the column mixture
    pub weights: Vec<f64>,
    /// The prior on the view CRP alphas
    pub view_alpha_prior: CrpPrior,
    /// The log likeihood of the data under the current assignment
    #[serde(default)]
    pub loglike: f64,
    /// The log prior likelihood of component parameters under the prior and of
    /// feature prior parameters under the hyperprior
    #[serde(default)]
    pub log_prior: f64,
    /// The log prior likelihood of the row assignments under CRP and of the CRP
    /// alpha param under the hyperprior
    #[serde(default)]
    pub log_view_alpha_prior: f64,
    /// The log prior likelihood of column assignment under CRP and of the state
    /// CRP alpha param under the hyperprior
    #[serde(default)]
    pub log_state_alpha_prior: f64,
    /// The running diagnostics
    pub diagnostics: StateDiagnostics,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}

impl State {
    pub fn new(
        views: Vec<View>,
        asgn: Assignment,
        view_alpha_prior: CrpPrior,
    ) -> Self {
        let weights = asgn.weights();

        let mut state = State {
            views,
            asgn,
            weights,
            view_alpha_prior,
            loglike: 0.0,
            log_prior: 0.0,
            log_state_alpha_prior: 0.0,
            log_view_alpha_prior: 0.0,
            diagnostics: StateDiagnostics::default(),
        };
        state.loglike = state.loglike();
        state
    }

    /// Draw a new `State` from the prior
    pub fn from_prior(
        mut ftrs: Vec<ColModel>,
        state_alpha_prior: CrpPrior,
        view_alpha_prior: CrpPrior,
        mut rng: &mut impl Rng,
    ) -> Self {
        let ncols = ftrs.len();
        let nrows = ftrs.get(0).map(|f| f.len()).unwrap_or(0);
        let asgn = AssignmentBuilder::new(ncols)
            .with_prior(state_alpha_prior)
            .seed_from_rng(&mut rng)
            .build()
            .unwrap();

        let mut views: Vec<View> = (0..asgn.ncats)
            .map(|_| {
                ViewBuilder::new(nrows)
                    .with_alpha_prior(view_alpha_prior.clone())
                    .seed_from_rng(&mut rng)
                    .build()
            })
            .collect();

        for (&v, ftr) in asgn.asgn.iter().zip(ftrs.drain(..)) {
            views[v].init_feature(ftr, &mut rng);
        }

        let weights = asgn.weights();

        let mut state = State {
            views,
            asgn,
            weights,
            view_alpha_prior,
            loglike: 0.0,
            log_prior: 0.0,
            log_state_alpha_prior: 0.0,
            log_view_alpha_prior: 0.0,
            diagnostics: StateDiagnostics::default(),
        };
        state.loglike = state.loglike();
        state
    }

    // Extend the columns by a number of cells, increasing the total number of
    // rows. The added entries will be empty.
    pub fn extend_cols(&mut self, nrows: usize) {
        self.views
            .iter_mut()
            .for_each(|view| view.extend_cols(nrows))
    }

    /// Mainly used for debugging. Always saves as yaml
    pub fn save(&mut self, dir: &Path, id: usize) -> io::Result<()> {
        save_state(dir, self, id, &FileConfig::default())
    }

    /// Get a reference to the features at `col_ix`
    #[inline]
    pub fn feature(&self, col_ix: usize) -> &ColModel {
        let view_ix = self.asgn.asgn[col_ix];
        &self.views[view_ix].ftrs[&col_ix]
    }

    /// Get a mutable reference to the features at `col_ix`
    #[inline]
    pub fn feature_mut(&mut self, col_ix: usize) -> &mut ColModel {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].ftrs.get_mut(&col_ix).unwrap()
    }

    /// Get a mixture model representation of the features at `col_ix`
    #[inline]
    pub fn feature_as_mixture(&self, col_ix: usize) -> MixtureType {
        let weights = {
            let view_ix = self.asgn.asgn[col_ix];
            self.views[view_ix].weights.clone()
        };
        self.feature(col_ix).to_mixture(weights)
    }

    /// Get the number of rows
    #[inline]
    pub fn nrows(&self) -> usize {
        self.views.get(0).map(|v| v.nrows()).unwrap_or(0)
    }

    /// Get the number of columns
    #[inline]
    pub fn ncols(&self) -> usize {
        self.views.iter().fold(0, |acc, v| acc + v.ncols())
    }

    /// Get the number of views
    #[inline]
    pub fn nviews(&self) -> usize {
        self.views.len()
    }

    /// Returns true if the State has no view, no rows, or no columns
    #[inline]
    pub fn is_empty(&self) -> bool {
        if self.views.is_empty() {
            true
        } else {
            self.ncols() == 0 || self.nrows() == 0
        }
    }

    /// Get the feature type (`FType`) of the column at `col_ix`
    #[inline]
    pub fn ftype(&self, col_ix: usize) -> FType {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].ftrs[&col_ix].ftype()
    }

    pub fn step(
        &mut self,
        transitions: &[StateTransition],
        mut rng: &mut impl Rng,
    ) {
        for transition in transitions {
            match transition {
                StateTransition::ColumnAssignment(alg) => {
                    self.reassign(*alg, transitions, &mut rng);
                }
                StateTransition::RowAssignment(alg) => {
                    self.reassign_rows(*alg, &mut rng);
                }
                StateTransition::StateAlpha => {
                    self.log_state_alpha_prior = self
                        .asgn
                        .update_alpha(braid_consts::MH_PRIOR_ITERS, &mut rng);
                }
                StateTransition::ViewAlphas => {
                    self.log_view_alpha_prior =
                        self.update_view_alphas(&mut rng);
                }
                StateTransition::FeaturePriors => {
                    self.log_prior = self.update_feature_priors(&mut rng);
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
        let mut rngs: Vec<Xoshiro256Plus> = (0..self.nviews())
            .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
            .collect();

        self.views.par_iter_mut().zip(rngs.par_iter_mut()).for_each(
            |(view, mut vrng)| {
                view.reassign(row_asgn_alg, &mut vrng);
            },
        );
    }

    #[inline]
    fn update_view_alphas(&mut self, mut rng: &mut impl Rng) -> f64 {
        self.views
            .iter_mut()
            .map(|v| v.update_alpha(&mut rng))
            .sum()
    }

    #[inline]
    fn update_feature_priors(&mut self, mut rng: &mut impl Rng) -> f64 {
        let mut rngs: Vec<Xoshiro256Plus> = (0..self.nviews())
            .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
            .collect();

        self.views
            .par_iter_mut()
            .zip(rngs.par_iter_mut())
            .map(|(v, mut trng)| v.update_prior_params(&mut trng))
            .sum()
    }

    #[inline]
    fn update_component_params(&mut self, mut rng: &mut impl Rng) {
        let mut rngs: Vec<_> = (0..self.nviews())
            .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
            .collect();

        self.views
            .par_iter_mut()
            .zip(rngs.par_iter_mut())
            .for_each(|(v, trng)| v.update_component_params(trng))
    }

    pub fn default_transitions() -> Vec<StateTransition> {
        // NOTE: we choose gibbs as the default algorithm because it is correct
        // and, unlike slice, it does not require consideration of the CRP prior
        // (slice can break if the alpha prior has infinite variance)
        vec![
            StateTransition::ColumnAssignment(ColAssignAlg::Gibbs),
            StateTransition::StateAlpha,
            StateTransition::RowAssignment(RowAssignAlg::Gibbs),
            StateTransition::ViewAlphas,
            StateTransition::FeaturePriors,
        ]
    }

    pub fn update(
        &mut self,
        config: StateUpdateConfig,
        mut rng: &mut impl Rng,
    ) {
        let time_started = Instant::now();
        for iter in 0..config.n_iters {
            self.step(&config.transitions, &mut rng);
            self.push_diagnostics();

            let duration = time_started.elapsed().as_secs();
            if config.check_complete(duration, iter) {
                break;
            }
        }
        self.finish_update(config.output_info)
            .expect("Failed to save");
    }

    fn finish_update(
        &mut self,
        output_info: Option<StateOutputInfo>,
    ) -> io::Result<()> {
        match output_info {
            Some(info) => {
                let path = info.path.as_path();
                path_validator(path).and_then(|_| self.save(path, info.id))
            }
            None => Ok(()),
        }
    }

    #[inline]
    fn push_diagnostics(&mut self) {
        // Sort the number of categories in each view
        let ncats = {
            let mut ncats: Vec<usize> =
                self.views.iter().map(|view| view.asgn.ncats).collect();

            ncats.sort();
            ncats
        };

        let nviews = ncats.len();
        let ncats_min = ncats[0];
        let ncats_max = ncats[nviews - 1];
        let ncats_median: f64 = if nviews == 1 {
            ncats[0] as f64
        } else if nviews % 2 == 0 {
            let split = nviews / 2;
            (ncats[split - 1] + ncats[split]) as f64 / 2.0
        } else {
            let split = nviews / 2;
            ncats[split] as f64
        };

        debug_assert!(ncats_min as f64 <= ncats_median);
        debug_assert!(ncats_median <= ncats_max as f64);

        self.diagnostics.loglike.push(self.loglike);
        self.diagnostics.nviews.push(self.asgn.ncats);
        self.diagnostics.state_alpha.push(self.asgn.alpha);

        self.diagnostics.ncats_median.push(ncats_median);
        self.diagnostics.ncats_min.push(ncats_min);
        self.diagnostics.ncats_max.push(ncats_max);

        let log_prior = self.log_prior
            + self.log_view_alpha_prior
            + self.log_state_alpha_prior;
        self.diagnostics.log_prior.push(log_prior);
    }

    // Reassign all columns to one view
    pub fn flatten_cols<R: rand::Rng>(&mut self, mut rng: &mut R) {
        let ncols = self.ncols();
        let new_asgn_vec = vec![0; ncols];
        let ncats = self.asgn.ncats;

        let ftrs = {
            let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
            for (i, &v) in self.asgn.asgn.iter().enumerate() {
                ftrs.push(
                    self.views[v].remove_feature(i).expect("Feature missing"),
                );
            }
            ftrs
        };

        self.integrate_finite_asgn(new_asgn_vec, ftrs, ncats, &mut rng);
        self.weights = vec![1.0];
    }

    pub fn reassign(
        &mut self,
        alg: ColAssignAlg,
        transitions: &[StateTransition],
        mut rng: &mut impl Rng,
    ) {
        match alg {
            ColAssignAlg::FiniteCpu => {
                self.reassign_cols_finite_cpu(transitions, &mut rng)
            }
            ColAssignAlg::Gibbs => {
                self.reassign_cols_gibbs(transitions, &mut rng)
            }
            ColAssignAlg::Slice => {
                self.reassign_cols_slice(transitions, &mut rng)
            }
        }
    }

    /// Insert new features into the `State`
    pub fn insert_new_features(
        &mut self,
        mut ftrs: Vec<ColModel>,
        mut rng: &mut impl Rng,
    ) {
        ftrs.drain(..)
            .map(|mut ftr| {
                if ftr.len() != self.nrows() {
                    panic!(
                        "State has {} rows, but feature has {}",
                        self.nrows(),
                        ftr.len()
                    );
                } else {
                    // increases as features inserted
                    ftr.set_id(self.ncols());
                    // do we always want draw_alpha to be true here?
                    self.insert_feature(ftr, true, &mut rng);
                }
            })
            .collect()
    }

    pub(crate) fn append_blank_features<R: Rng>(
        &mut self,
        mut ftrs: Vec<ColModel>,
        mut rng: &mut R,
    ) {
        use rv::misc::pflip;

        if self.nviews() == 0 {
            self.views.push(ViewBuilder::new(0).build())
        }

        let k = self.nviews();
        let p = (k as f64).recip();
        ftrs.drain(..).for_each(|mut ftr| {
            ftr.set_id(self.ncols());
            self.asgn.push_unassigned();
            // insert into random existing view
            let view_ix = pflip(&vec![p; k], 1, &mut rng)[0];
            self.asgn.reassign(self.ncols(), view_ix);
            self.views[view_ix].insert_feature(ftr, &mut rng);
        })
    }

    // Finds all unassigned rows in each view and reassigns them
    pub(crate) fn assign_unassigned<R: Rng>(&mut self, mut rng: &mut R) {
        self.views
            .iter_mut()
            .for_each(|view| view.assign_unassigned(&mut rng));
    }

    /// Insert an unassigned feature into the `State` via the `Gibbs`
    /// algorithm. If the feature is new, it is appended to the end of the
    /// `State`.
    pub fn insert_feature(
        &mut self,
        ftr: ColModel,
        draw_alpha: bool,
        mut rng: &mut impl Rng,
    ) -> f64 {
        // Number of singleton features. For assigning to a singleton, we have
        // to estimate the marginal likelihood via Monte Carlo integration. The
        // `m` parameter is the number of samples for the integration.
        let m: usize = 3; // TODO: Should this be a parameter in ColAssignAlg?
        let col_ix = ftr.id();

        // crp alpha divided by the number of MC samples
        let a_part = (self.asgn.alpha / m as f64).ln();

        // score for each view. We will push the singleton view probabilities
        // later
        let mut logps = self.asgn.log_dirvec(false);

        // maintain a vec that  holds just the likelihoods
        let mut ftr_logps: Vec<f64> = Vec::with_capacity(logps.len());

        // TODO: might be faster with an iterator?
        for (ix, view) in self.views.iter().enumerate() {
            let lp = ftr.asgn_score(&view.asgn);
            ftr_logps.push(lp);
            logps[ix] += lp;
        }

        let nviews = self.nviews();

        // here we create the monte carlo estimate for the singleton view
        let mut tmp_asgns: BTreeMap<usize, Assignment> = (0..m)
            .map(|i| {
                // assignment for a hypothetical singleton view
                let asgn_bldr = AssignmentBuilder::new(self.nrows())
                    .with_prior(self.view_alpha_prior.clone());

                // If we do not want to draw a view alpha, take an existing one from the
                // first view. This covers the case were we set the view alphas and
                // never transitions them, for example if we are doing geweke on a
                // subset of transitions.
                let tmp_asgn = if draw_alpha {
                    asgn_bldr.seed_from_rng(&mut rng).build().unwrap()
                } else {
                    let alpha = self.views[0].asgn.alpha;
                    asgn_bldr
                        .with_alpha(alpha)
                        .seed_from_rng(&mut rng)
                        .build()
                        .unwrap()
                };

                // log likelihood of singleton feature
                // TODO: add `m` in {1, 2, ...} parameter that dictates how many
                // singletons to try.
                let singleton_logp = ftr.asgn_score(&tmp_asgn);
                ftr_logps.push(singleton_logp);
                logps.push(a_part + singleton_logp);

                (i + nviews, tmp_asgn)
            })
            .collect();

        debug_assert_eq!(nviews + m, logps.len());

        // Gibbs step (draw from categorical)
        let v_new = ln_pflip(&logps, 1, false, &mut rng)[0];
        let logp_out = ftr_logps[v_new];

        // If we chose a singleton view...
        if v_new >= nviews {
            // This will error if v_new is not in the index, and that is a good.
            // thing.
            let tmp_asgn = tmp_asgns.remove(&v_new).unwrap();
            let new_view = ViewBuilder::from_assignment(tmp_asgn)
                .seed_from_rng(&mut rng)
                .build();
            self.views.push(new_view);
        }

        // If v_new is >= nviews, it means that we chose the singleton view, so
        // we max the new view index to nviews
        let v_new = v_new.min(nviews);

        self.asgn.reassign(col_ix, v_new);
        self.views[v_new].insert_feature(ftr, &mut rng);
        logp_out
    }

    #[inline]
    pub fn reassign_col_gibbs(
        &mut self,
        col_ix: usize,
        draw_alpha: bool,
        mut rng: &mut impl Rng,
    ) -> f64 {
        let ftr = self.extract_ftr(col_ix);
        self.insert_feature(ftr, draw_alpha, &mut rng)
    }

    /// Reassign all columns using the Gibbs transition.
    ///
    /// # Notes
    /// The transitions are passed to ensure that Geweke tests on subsets of
    /// transitions will still pass. For example, if we are not doing the
    /// `ViewAlpha` transition, we should not draw an alpha from the prior for
    /// the singleton view; instead we should use the existing view alpha.
    pub fn reassign_cols_gibbs(
        &mut self,
        transitions: &[StateTransition],
        mut rng: &mut impl Rng,
    ) {
        if self.ncols() == 1 {
            return;
        }
        // The algorithm is not valid if the columns are not scanned in
        // random order
        let draw_alpha = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewAlphas);

        let mut col_ixs: Vec<usize> = (0..self.ncols()).map(|i| i).collect();
        col_ixs.shuffle(&mut rng);

        self.loglike = col_ixs
            .drain(..)
            .map(|col_ix| self.reassign_col_gibbs(col_ix, draw_alpha, &mut rng))
            .sum::<f64>();

        // NOTE: The oracle functions use the weights to compute probabilities.
        // Since the Gibbs algorithm uses implicit weights from the partition,
        // it does not explicitly update the weights. Non-updated weights means
        // wrong probabilities. To avoid this, we set the weights by the
        // partition here.
        self.weights = self.asgn.weights();
    }

    /// Reassign columns to views using the `FiniteCpu` transition
    pub fn reassign_cols_finite_cpu(
        &mut self,
        transitions: &[StateTransition],
        mut rng: &mut impl Rng,
    ) {
        let ncols = self.ncols();

        if ncols == 1 {
            return;
        }

        let draw_alpha = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewAlphas);
        self.resample_weights(true, &mut rng);
        self.append_empty_view(draw_alpha, &mut rng);

        let log_weights: Vec<f64> =
            self.weights.iter().map(|w| w.ln()).collect();
        let ncats = self.asgn.ncats + 1;

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
        for (i, &v) in self.asgn.asgn.iter().enumerate() {
            ftrs.push(
                self.views[v].remove_feature(i).expect("Feature missing"),
            );
        }

        let logps = {
            let values: Vec<f64> = ftrs
                .par_iter()
                .flat_map(|ftr| {
                    self.views
                        .iter()
                        .enumerate()
                        .map(|(v, view)| {
                            ftr.asgn_score(&view.asgn) + log_weights[v]
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();

            Matrix::from_raw_parts(values, ftrs.len())
        };

        let new_asgn_vec = massflip(&logps, &mut rng);

        self.loglike = new_asgn_vec
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, z)| acc + logps[(i, *z)]);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, ncats, &mut rng);
        self.resample_weights(false, &mut rng);
    }

    /// Reassign columns to views using the improved slice sampler
    pub fn reassign_cols_slice(
        &mut self,
        transitions: &[StateTransition],
        mut rng: &mut impl Rng,
    ) {
        use crate::dist::stick_breaking::sb_slice_extend;

        if self.ncols() == 1 {
            return;
        }

        self.resample_weights(false, &mut rng);

        let ncols = self.ncols();

        let udist = rand::distributions::Open01;

        let weights: Vec<f64> = {
            let dirvec = self.asgn.dirvec(true);
            let dir = Dirichlet::new(dirvec).unwrap();
            dir.draw(&mut rng)
        };

        let us: Vec<f64> = self
            .asgn
            .asgn
            .iter()
            .map(|&zi| {
                let wi: f64 = weights[zi];
                let u: f64 = rng.sample(udist);
                u * wi
            })
            .collect();

        let u_star: f64 =
            us.iter()
                .fold(1.0, |umin, &ui| if ui < umin { ui } else { umin });

        // Variable shadowing
        let weights =
            sb_slice_extend(weights, self.asgn.alpha, u_star, &mut rng)
                .unwrap();

        let n_new_views = weights.len() - self.weights.len();
        let nviews = weights.len();

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(ncols);
        for (i, &v) in self.asgn.asgn.iter().enumerate() {
            ftrs.push(
                self.views[v].remove_feature(i).expect("Feature missing"),
            );
        }

        let draw_alpha = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewAlphas);
        for _ in 0..n_new_views {
            self.append_empty_view(draw_alpha, &mut rng);
        }

        // initialize truncated log probabilities
        let logps = {
            let values: Vec<f64> = ftrs
                .par_iter()
                .zip(us.par_iter())
                .flat_map(|(ftr, ui)| {
                    self.views
                        .iter()
                        .zip(weights.iter())
                        .map(|(view, w)| {
                            if w >= ui {
                                ftr.asgn_score(&view.asgn)
                            } else {
                                NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();

            Matrix::from_raw_parts(values, ftrs.len())
        };

        let new_asgn_vec = massflip_slice_mat_par(&logps, &mut rng);

        self.loglike = {
            let log_weights: Vec<f64> =
                weights.iter().map(|w| (*w).ln()).collect();

            new_asgn_vec
                .iter()
                .enumerate()
                .fold(0.0, |acc, (i, z)| acc + logps[(i, *z)] + log_weights[*z])
        };

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

    #[inline]
    pub fn logp_at(&self, row_ix: usize, col_ix: usize) -> Option<f64> {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].logp_at(row_ix, col_ix)
    }

    #[inline]
    pub fn datum(&self, row_ix: usize, col_ix: usize) -> Datum {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].datum(row_ix, col_ix).unwrap()
    }

    pub fn resample_weights(
        &mut self,
        add_empty_component: bool,
        mut rng: &mut impl Rng,
    ) {
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

        self.asgn
            .set_asgn(new_asgn_vec)
            .expect("new_asgn_vec is invalid");

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

    pub fn component(&self, row_ix: usize, col_ix: usize) -> Component {
        let view_ix = self.asgn.asgn[col_ix];
        let view = &self.views[view_ix];
        let k = view.asgn.asgn[row_ix];
        view.ftrs[&col_ix].component(k)
    }

    /// Remove the view, but do not adjust any other metadata
    #[inline]
    fn drop_view(&mut self, view_ix: usize) {
        // view goes out of scope and is dropped
        let _view = self.views.remove(view_ix);
    }

    fn append_empty_view(
        &mut self,
        draw_alpha: bool, // draw the view CRP alpha from the prior
        mut rng: &mut impl Rng,
    ) {
        let asgn_builder = AssignmentBuilder::new(self.nrows())
            .with_prior(self.view_alpha_prior.clone());

        let asgn_builder = if draw_alpha {
            asgn_builder
        } else {
            // The alphas should all be the same, so just take one from another view
            let alpha = self.views[0].asgn.alpha;
            asgn_builder.with_alpha(alpha)
        };

        let asgn = asgn_builder.seed_from_rng(&mut rng).build().unwrap();

        let view = ViewBuilder::from_assignment(asgn)
            .seed_from_rng(&mut rng)
            .build();

        self.views.push(view)
    }

    #[inline]
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

    pub fn insert_datum(&mut self, row_ix: usize, col_ix: usize, x: Datum) {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].insert_datum(row_ix, col_ix, x);
    }

    pub fn drop_data(&mut self) {
        let _data = self.take_data();
    }

    // Delete the top/front n rows.
    pub fn del_rows_at(&mut self, ix: usize, n: usize) {
        self.views
            .iter_mut()
            .for_each(|view| view.del_rows_at(ix, n));
    }

    pub fn repop_data(&mut self, mut data: BTreeMap<usize, FeatureData>) {
        if data.len() != self.ncols() {
            panic!("Data length and state.ncols differ");
        } else if (0..self.ncols()).any(|k| !data.contains_key(&k)) {
            panic!("Data does not contain all column IDs");
        } else {
            let ids: Vec<usize> = data.keys().copied().collect();
            for id in ids {
                let data_col = data.remove(&id).unwrap();
                self.feature_mut(id).repop_data(data_col);
            }
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

    // Forget and re-observe all the data.
    // since the data change during the gewek posterior chain runs, the
    // suffstats get out of wack, so we need to re-obseve the new data.
    fn refresh_suffstats(&mut self, mut rng: &mut impl Rng) {
        self.views
            .iter_mut()
            .for_each(|v| v.refresh_suffstats(&mut rng));
    }

    pub fn col_weights(&self, col_ix: usize) -> Vec<f64> {
        let view_ix = self.asgn.asgn[col_ix];
        self.views[view_ix].asgn.weights()
    }

    // // FIXME: implment MixtueType::from(ColModel) instead
    // pub fn feature_as_mixture(&self, col_ix: usize) -> MixtureType {
    //     self.feature(col_ix).to_mixture()
    // }
}

// Geweke
// ======
use crate::cc::feature::geweke::gen_geweke_col_models;
use braid_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct StateGewekeSettings {
    /// The number of columns/features in the state
    pub ncols: usize,
    /// The number of rows in the state
    pub nrows: usize,
    /// Column Model types
    pub cm_types: Vec<FType>,
    /// Which transitions to do
    pub transitions: Vec<StateTransition>,
}

impl StateGewekeSettings {
    pub fn new(nrows: usize, cm_types: Vec<FType>) -> Self {
        StateGewekeSettings {
            ncols: cm_types.len(),
            nrows,
            cm_types,
            transitions: State::default_transitions(),
        }
    }

    pub fn do_col_asgn_transition(&self) -> bool {
        self.transitions.iter().any(|t| match t {
            StateTransition::ColumnAssignment(_) => true,
            _ => false,
        })
    }

    pub fn do_row_asgn_transition(&self) -> bool {
        self.transitions.iter().any(|t| match t {
            StateTransition::RowAssignment(_) => true,
            _ => false,
        })
    }

    pub fn do_alpha_transition(&self) -> bool {
        self.transitions.iter().any(|t| match t {
            StateTransition::StateAlpha => true,
            _ => false,
        })
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
            cm_types: vec![],
            transitions: s
                .transitions
                .iter()
                .filter_map(|&st| st.try_into().ok())
                .collect(),
        };
        for view in &mut self.views {
            view.geweke_resample_data(Some(&view_settings), &mut rng);
        }
    }
}

/// The State summary for Geweke
#[derive(Clone, Debug)]
pub struct GewekeStateSummary {
    /// The number of views
    pub nviews: Option<usize>,
    /// CRP alpha
    pub alpha: Option<f64>,
    /// The summary for each view
    pub views: Vec<GewekeViewSummary>,
}

impl From<&GewekeStateSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeStateSummary) -> Self {
        let mut map: BTreeMap<String, f64> = BTreeMap::new();

        if let Some(nviews) = value.nviews {
            map.insert("n views".into(), nviews as f64);
        }

        if let Some(alpha) = value.alpha {
            map.insert("crp alpha".into(), alpha);
        }

        value.views.iter().for_each(|view_summary| {
            let view_map: BTreeMap<String, f64> = view_summary.into();
            view_map.iter().for_each(|(key, val)| {
                map.insert(key.clone(), *val);
            });
        });
        map
    }
}

impl From<GewekeStateSummary> for BTreeMap<String, f64> {
    fn from(value: GewekeStateSummary) -> Self {
        Self::from(&value)
    }
}

impl GewekeSummarize for State {
    type Summary = GewekeStateSummary;

    fn geweke_summarize(
        &self,
        settings: &StateGewekeSettings,
    ) -> GewekeStateSummary {
        // Dummy settings. the only thing the view summarizer cares about is the
        // transitions.
        let view_settings = ViewGewekeSettings {
            ncols: 0,
            nrows: 0,
            cm_types: vec![],
            transitions: settings
                .transitions
                .iter()
                .filter_map(|&st| st.try_into().ok())
                .collect(),
        };

        GewekeStateSummary {
            nviews: if settings.do_col_asgn_transition() {
                Some(self.asgn.ncats)
            } else {
                None
            },
            alpha: if settings.do_alpha_transition() {
                Some(self.asgn.alpha)
            } else {
                None
            },
            views: self
                .views
                .iter()
                .map(|view| view.geweke_summarize(&view_settings))
                .collect(),
        }
    }
}

// XXX: Note that the only Geweke is only guaranteed to return turn results if
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

        let do_col_asgn_transition = settings.do_col_asgn_transition();
        let do_row_asgn_transition = settings.do_row_asgn_transition();

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
        }
        .seed_from_rng(&mut rng)
        .with_geweke_prior();

        let asgn = if do_state_alpha_transition {
            asgn_bldr.build().unwrap()
        } else {
            asgn_bldr.with_alpha(1.0).build().unwrap()
        };

        #[allow(clippy::collapsible_if)]
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
        }
        .with_geweke_prior();

        let mut views: Vec<View> = (0..asgn.ncats)
            .map(|_| {
                let asgn = view_asgn_bldr
                    .clone()
                    .seed_from_rng(&mut rng)
                    .build()
                    .unwrap();
                ViewBuilder::from_assignment(asgn)
                    .seed_from_rng(&mut rng)
                    .build()
            })
            .collect();

        for (&v, ftr) in asgn.asgn.iter().zip(ftrs.drain(..)) {
            views[v].init_feature(ftr, &mut rng);
        }

        let view_alpha_prior = views[0].asgn.prior.clone();

        let weights = asgn.weights();
        State {
            views,
            asgn,
            weights,
            view_alpha_prior,
            loglike: 0.0,
            log_prior: 0.0,
            log_state_alpha_prior: 0.0,
            log_view_alpha_prior: 0.0,
            diagnostics: StateDiagnostics::default(),
        }
    }

    fn geweke_step(
        &mut self,
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        let config = StateUpdateConfig {
            transitions: settings.transitions.clone(),
            n_iters: 1,
            ..Default::default()
        };

        self.refresh_suffstats(&mut rng);
        self.update(config, &mut rng);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::{fs::remove_dir_all, path::Path};

    use crate::benchmark::StateBuilder;
    use approx::*;
    use braid_codebook::ColType;

    #[test]
    fn extract_ftr_non_singleton() {
        let mut state = StateBuilder::new()
            .with_rows(50)
            .add_column_configs(
                4,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .with_views(2)
            .build()
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
        let mut state = StateBuilder::new()
            .with_rows(50)
            .add_column_configs(
                3,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .with_views(2)
            .build()
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
            .add_column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .with_views(4)
            .with_cats(5)
            .build()
            .expect("Failed to build state");

        let config = StateUpdateConfig {
            n_iters: 100,
            transitions: vec![StateTransition::ColumnAssignment(
                ColAssignAlg::Gibbs,
            )],
            ..Default::default()
        };

        state.update(config, &mut rng);
    }

    #[test]
    fn gibbs_row_transition_smoke() {
        let mut rng = rand::thread_rng();
        let mut state = StateBuilder::new()
            .with_rows(10)
            .add_column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .with_views(4)
            .with_cats(5)
            .build()
            .expect("Failed to build state");

        let config = StateUpdateConfig {
            n_iters: 100,
            transitions: vec![StateTransition::RowAssignment(
                RowAssignAlg::Gibbs,
            )],
            ..Default::default()
        };
        state.update(config, &mut rng);
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
            if let CrpPrior::Gamma(gamma) = state.asgn.prior {
                assert_relative_eq!(gamma.shape(), 3.0, epsilon = 1E-12);
                assert_relative_eq!(gamma.rate(), 3.0, epsilon = 1E-12);
            } else {
                panic!("State alpha prior was not gamma")
            }
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
                if let CrpPrior::Gamma(gamma) = &view.asgn.prior {
                    assert_relative_eq!(gamma.shape(), 3.0, epsilon = 1E-12);
                    assert_relative_eq!(gamma.rate(), 3.0, epsilon = 1E-12);
                } else {
                    panic!("State alpha prior was not gamma")
                }
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
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
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
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
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
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
                StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
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

    #[test]
    fn update_timeout_should_stop_update() {
        let mut rng = rand::thread_rng();

        let n_iters = 1_000_000; // should not get done in 2 sec
        let config = StateUpdateConfig {
            n_iters,
            timeout: Some(2),
            ..Default::default()
        };

        let colmd = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        let mut state = StateBuilder::new()
            .add_column_configs(10, colmd)
            .with_rows(1000)
            .build()
            .unwrap();

        let time_started = Instant::now();
        state.update(config, &mut rng);
        let elapsed = time_started.elapsed().as_secs();

        assert!(2 <= elapsed && elapsed <= 3);
        assert!(state.diagnostics.loglike.len() < n_iters);
    }

    #[test]
    fn update_should_stop_at_max_iters() {
        let mut rng = rand::thread_rng();

        let n_iters = 37;
        let config = StateUpdateConfig {
            n_iters,
            timeout: Some(86_400),
            ..Default::default()
        };

        let colmd = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        let mut state = StateBuilder::new()
            .add_column_configs(10, colmd)
            .with_rows(1000)
            .build()
            .unwrap();

        state.update(config, &mut rng);

        assert_eq!(state.diagnostics.loglike.len(), n_iters);
    }

    #[test]
    fn state_save_after_run_if_requested() {
        let mut rng = rand::thread_rng();
        let dir = String::from("delete_me.braidtrash");
        let config = StateUpdateConfig {
            n_iters: 10,
            output_info: Some(StateOutputInfo::new(dir.clone(), 0)),
            ..Default::default()
        };

        let colmd = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        let mut state = StateBuilder::new()
            .add_column_configs(10, colmd)
            .with_rows(1000)
            .build()
            .unwrap();

        state.update(config, &mut rng);

        let state_fname = format!("{}/0.state", dir);
        let state_path = Path::new(state_fname.as_str());
        let state_saved = state_path.exists();

        remove_dir_all(Path::new(dir.as_str())).expect("Cleanup failed");

        assert!(state_saved);
    }

    #[test]
    fn flatten_cols() {
        let mut rng = rand::thread_rng();
        let colmd = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        let mut state = StateBuilder::new()
            .add_column_configs(20, colmd)
            .with_rows(10)
            .with_views(5)
            .build()
            .unwrap();

        assert_eq!(state.nviews(), 5);
        assert_eq!(state.ncols(), 20);

        state.flatten_cols(&mut rng);
        assert_eq!(state.nviews(), 1);
        assert_eq!(state.ncols(), 20);
    }
}

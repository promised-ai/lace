mod builder;
pub use builder::{BuildStateError, Builder};

use lace_data::{Datum, FeatureData};
use lace_stats::assignment::Assignment;
use lace_stats::prior_process::Builder as AssignmentBuilder;
use lace_stats::prior_process::{PriorProcess, PriorProcessT, Process};
use lace_stats::rv::dist::Dirichlet;
use lace_stats::rv::misc::ln_pflip;
use lace_stats::rv::traits::*;
use lace_stats::MixtureType;
use lace_utils::Matrix;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::alg::{ColAssignAlg, RowAssignAlg};
use crate::config::StateUpdateConfig;
use crate::feature::Component;
use crate::feature::{ColModel, FType, Feature};
use crate::transition::StateTransition;
use crate::view;
use crate::view::View;

pub mod geweke;
mod parallel_gibbs;
mod serial_gibbs;
mod slice;

/// Stores some diagnostic info in the `State` at every iteration
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
#[serde(default)]
pub struct StateDiagnostics {
    /// Log likelihood
    #[serde(default)]
    pub loglike: Vec<f64>,
    /// Log prior likelihood
    #[serde(default)]
    pub logprior: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct StateScoreComponents {
    pub ln_likelihood: f64,
    pub ln_prior: f64,
    pub ln_state_prior_process: f64,
    pub ln_view_prior_process: f64,
}

/// A cross-categorization state
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct State {
    /// The views of columns
    pub views: Vec<View>,
    /// The assignment of columns to views
    pub prior_process: PriorProcess,
    /// The weights of each view in the column mixture
    pub weights: Vec<f64>,
    #[serde(default)]
    pub score: StateScoreComponents,
    /// The running diagnostics
    pub diagnostics: StateDiagnostics,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}

impl State {
    pub fn new(views: Vec<View>, prior_process: PriorProcess) -> Self {
        let weights = prior_process.weight_vec(false);

        let mut state = State {
            views,
            prior_process,
            weights,
            score: StateScoreComponents::default(),
            diagnostics: StateDiagnostics::default(),
        };
        state.score.ln_likelihood = state.loglike();
        state
    }

    pub fn asgn(&self) -> &Assignment {
        &self.prior_process.asgn
    }

    pub fn asgn_mut(&mut self) -> &mut Assignment {
        &mut self.prior_process.asgn
    }

    /// Create a new `Builder` for generating a new `State`.
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Draw a new `State` from the prior
    pub fn from_prior<R: Rng>(
        mut ftrs: Vec<ColModel>,
        state_process: Process,
        view_process: Process,
        rng: &mut R,
    ) -> Self {
        let n_cols = ftrs.len();
        let n_rows = ftrs.first().map(|f| f.len()).unwrap_or(0);
        let prior_process =
            PriorProcess::from_process(state_process, n_cols, rng);
        let mut views: Vec<View> = (0..prior_process.asgn.n_cats)
            .map(|_| {
                view::Builder::new(n_rows)
                    .prior_process(view_process.clone())
                    .seed_from_rng(rng)
                    .build()
            })
            .collect();

        // TODO: Can we parallellize this?
        for (&v, ftr) in prior_process.asgn.iter().zip(ftrs.drain(..)) {
            views[v].init_feature(ftr, rng);
        }

        let weights = prior_process.weight_vec(false);

        let mut state = State {
            views,
            prior_process,
            weights,
            score: StateScoreComponents::default(),
            diagnostics: StateDiagnostics::default(),
        };
        state.score.ln_likelihood = state.loglike();
        state
    }

    // Extend the columns by a number of cells, increasing the total number of
    // rows. The added entries will be empty.
    pub fn extend_cols(&mut self, n_rows: usize) {
        self.views
            .iter_mut()
            .for_each(|view| view.extend_cols(n_rows))
    }

    /// Get a reference to the features at `col_ix`
    #[inline]
    pub fn feature(&self, col_ix: usize) -> &ColModel {
        let view_ix = self.asgn().asgn[col_ix];
        &self.views[view_ix].ftrs[&col_ix]
    }

    /// Get a mutable reference to the features at `col_ix`
    #[inline]
    pub fn feature_mut(&mut self, col_ix: usize) -> &mut ColModel {
        let view_ix = self.asgn().asgn[col_ix];
        self.views[view_ix].ftrs.get_mut(&col_ix).unwrap()
    }

    /// Get a mixture model representation of the features at `col_ix`
    #[inline]
    pub fn feature_as_mixture(&self, col_ix: usize) -> MixtureType {
        let weights = {
            let view_ix = self.asgn().asgn[col_ix];
            self.views[view_ix].weights.clone()
        };
        self.feature(col_ix).to_mixture(weights)
    }

    /// Get the number of rows
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.views.first().map(|v| v.n_rows()).unwrap_or(0)
    }

    /// Get the number of columns
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.views.iter().fold(0, |acc, v| acc + v.n_cols())
    }

    /// Get the number of views
    #[inline]
    pub fn n_views(&self) -> usize {
        self.views.len()
    }

    /// Returns true if the State has no view, no rows, or no columns
    #[inline]
    pub fn is_empty(&self) -> bool {
        if self.views.is_empty() {
            true
        } else {
            self.n_cols() == 0 || self.n_rows() == 0
        }
    }

    /// Get the feature type (`FType`) of the column at `col_ix`
    #[inline]
    pub fn ftype(&self, col_ix: usize) -> FType {
        let view_ix = self.asgn().asgn[col_ix];
        self.views[view_ix].ftrs[&col_ix].ftype()
    }

    pub fn step<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        rng: &mut R,
    ) {
        for transition in transitions {
            match transition {
                StateTransition::ColumnAssignment(alg) => {
                    self.reassign(*alg, transitions, rng);
                }
                StateTransition::RowAssignment(alg) => {
                    self.reassign_rows(*alg, rng);
                }
                StateTransition::StatePriorProcessParams => {
                    // FIXME: Add to probability?
                    self.score.ln_state_prior_process =
                        self.prior_process.update_params(rng);
                }
                StateTransition::ViewPriorProcessParams => {
                    self.score.ln_view_prior_process =
                        self.update_view_prior_process_params(rng);
                }
                StateTransition::FeaturePriors => {
                    self.score.ln_prior = self.update_feature_priors(rng);
                }
                StateTransition::ComponentParams => {
                    self.update_component_params(rng);
                }
            }
        }
        self.score.ln_likelihood = self.loglike();
    }

    fn reassign_rows<R: Rng>(
        &mut self,
        row_asgn_alg: RowAssignAlg,
        mut rng: &mut R,
    ) {
        let mut rngs: Vec<Xoshiro256Plus> = (0..self.n_views())
            .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
            .collect();

        self.views
            .par_iter_mut()
            .zip_eq(rngs.par_iter_mut())
            .for_each(|(view, mut t_rng)| {
                view.reassign(row_asgn_alg, &mut t_rng);
            });
    }

    #[inline]
    fn update_view_prior_process_params<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.views
            .iter_mut()
            .map(|v| v.update_prior_process_params(rng))
            .sum()
    }

    #[inline]
    fn update_feature_priors<R: Rng>(&mut self, mut rng: &mut R) -> f64 {
        let mut rngs: Vec<Xoshiro256Plus> = (0..self.n_views())
            .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
            .collect();

        self.views
            .par_iter_mut()
            .zip_eq(rngs.par_iter_mut())
            .map(|(v, t_rng)| v.update_prior_params(t_rng))
            .sum()
    }

    #[inline]
    fn update_component_params<R: Rng>(&mut self, mut rng: &mut R) {
        let mut rngs: Vec<_> = (0..self.n_views())
            .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
            .collect();

        self.views
            .par_iter_mut()
            .zip_eq(rngs.par_iter_mut())
            .for_each(|(v, t_rng)| v.update_component_params(t_rng))
    }

    pub fn update<R: Rng>(&mut self, config: StateUpdateConfig, rng: &mut R) {
        for iter in 0..config.n_iters {
            self.step(&config.transitions, rng);
            self.push_diagnostics();

            if config.check_over_iters(iter) {
                break;
            }
        }
    }

    pub fn push_diagnostics(&mut self) {
        self.diagnostics.loglike.push(self.score.ln_likelihood);
        let log_prior = self.score.ln_prior
            + self.score.ln_view_prior_process
            + self.score.ln_state_prior_process;
        self.diagnostics.logprior.push(log_prior);
    }

    // Reassign all columns to one view
    pub fn flatten_cols<R: rand::Rng>(&mut self, mut rng: &mut R) {
        let n_cols = self.n_cols();
        let new_asgn_vec = vec![0; n_cols];
        let n_cats = self.asgn().n_cats;

        let ftrs = {
            let mut ftrs: Vec<ColModel> = Vec::with_capacity(n_cols);
            for (i, &v) in self.prior_process.asgn.asgn.iter().enumerate() {
                ftrs.push(
                    self.views[v].remove_feature(i).expect("Feature missing"),
                );
            }
            ftrs
        };

        self.integrate_finite_asgn(new_asgn_vec, ftrs, n_cats, &mut rng);
        self.weights = vec![1.0];
    }

    pub fn reassign<R: Rng>(
        &mut self,
        alg: ColAssignAlg,
        transitions: &[StateTransition],
        rng: &mut R,
    ) {
        match alg {
            ColAssignAlg::FiniteCpu => {
                self.reassign_cols_finite_cpu(transitions, rng)
            }
            ColAssignAlg::Gibbs => {
                self.reassign_cols_gibbs(transitions, rng);
                // // FIXME: below alg doesn't pass enum tests
                // self.reassign_cols_gibbs_precomputed(transitions, rng);

                // NOTE: The oracle functions use the weights to compute probabilities.
                // Since the Gibbs algorithm uses implicit weights from the partition,
                // it does not explicitly update the weights. Non-updated weights means
                // wrong probabilities. To avoid this, we set the weights by the
                // partition here.
                self.weights = self.prior_process.weight_vec(false);
            }
            ColAssignAlg::Slice => self.reassign_cols_slice(transitions, rng),
        }
    }

    /// Insert new features into the `State`
    pub fn insert_new_features<R: Rng>(
        &mut self,
        mut ftrs: Vec<ColModel>,
        rng: &mut R,
    ) {
        ftrs.drain(..)
            .map(|mut ftr| {
                if ftr.len() != self.n_rows() {
                    panic!(
                        "State has {} rows, but feature has {}",
                        self.n_rows(),
                        ftr.len()
                    );
                } else {
                    // increases as features inserted
                    ftr.set_id(self.n_cols());
                    // do we always want draw_alpha to be true here?
                    self.insert_feature(ftr, true, rng);
                }
            })
            .collect()
    }

    pub fn append_blank_features<R: Rng>(
        &mut self,
        mut ftrs: Vec<ColModel>,
        mut rng: &mut R,
    ) {
        use lace_stats::rv::misc::pflip;

        if self.n_views() == 0 {
            self.views.push(view::Builder::new(0).build())
        }

        let k = self.n_views();
        let p = (k as f64).recip();
        ftrs.drain(..).for_each(|mut ftr| {
            ftr.set_id(self.n_cols());
            self.asgn_mut().push_unassigned();
            // insert into random existing view
            let view_ix = pflip(&vec![p; k], 1, &mut rng)[0];
            let n_cols = self.n_cols();
            self.asgn_mut().reassign(n_cols, view_ix);
            self.views[view_ix].insert_feature(ftr, &mut rng);
        })
    }

    // Finds all unassigned rows in each view and reassigns them
    pub fn assign_unassigned<R: Rng>(&mut self, mut rng: &mut R) {
        self.views
            .iter_mut()
            .for_each(|view| view.assign_unassigned(&mut rng));
    }

    fn create_tmp_assign(
        &self,
        draw_process_params: bool,
        seed: u64,
    ) -> PriorProcess {
        // assignment for a hypothetical singleton view
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let asgn_bldr =
            AssignmentBuilder::new(self.n_rows()).with_seed(rng.gen());
        // If we do not want to draw a view process params, take an
        // existing process from the first view. This covers the case
        // where we set the view process params and never transitions
        // them, for example if we are doing geweke on a subset of
        // transitions.
        let mut process = self.views[0].prior_process.process.clone();
        if draw_process_params {
            process.reset_params(&mut rng);
        };
        asgn_bldr.with_process(process).build().unwrap()
    }

    fn create_tmp_assigns(
        &self,
        counter_start: usize,
        draw_process_params: bool,
        seeds: &[u64],
    ) -> BTreeMap<usize, PriorProcess> {
        seeds
            .iter()
            .enumerate()
            .map(|(i, &seed)| {
                let tmp_asgn =
                    self.create_tmp_assign(draw_process_params, seed);

                (i + counter_start, tmp_asgn)
            })
            .collect()
    }

    /// Insert an unassigned feature into the `State` via the `Gibbs`
    /// algorithm. If the feature is new, it is appended to the end of the
    /// `State`.
    pub fn insert_feature<R: Rng>(
        &mut self,
        ftr: ColModel,
        update_process_params: bool,
        rng: &mut R,
    ) -> f64 {
        // Number of singleton features. For assigning to a singleton, we have
        // to estimate the marginal likelihood via Monte Carlo integration. The
        // `m` parameter is the number of samples for the integration.
        let m: usize = 1; // TODO: Should this be a parameter in ColAssignAlg?
        let col_ix = ftr.id();
        let n_views = self.n_views();

        // singleton weight divided by the number of MC samples
        let p_singleton =
            self.prior_process.process.ln_singleton_weight(n_views)
                - (m as f64).ln();

        // score for each view. We will push the singleton view probabilities
        // later
        let mut logps = self
            .asgn()
            .counts
            .iter()
            .map(|&n_k| self.prior_process.process.ln_gibbs_weight(n_k))
            .collect::<Vec<f64>>();

        // maintain a vec that  holds just the likelihoods
        let mut ftr_logps: Vec<f64> = Vec::with_capacity(logps.len());

        // TODO: might be faster with an iterator?
        for (ix, view) in self.views.iter().enumerate() {
            let lp = ftr.asgn_score(view.asgn());
            ftr_logps.push(lp);
            logps[ix] += lp;
        }

        // here we create the monte carlo estimate for the singleton view
        let mut tmp_asgns = {
            let seeds: Vec<u64> = (0..m).map(|_| rng.gen()).collect();
            self.create_tmp_assigns(n_views, update_process_params, &seeds)
        };

        tmp_asgns.iter().for_each(|(_, tmp_asgn)| {
            let singleton_logp = ftr.asgn_score(&tmp_asgn.asgn);
            ftr_logps.push(singleton_logp);
            logps.push(p_singleton + singleton_logp);
        });

        debug_assert_eq!(n_views + m, logps.len());

        // Gibbs step (draw from categorical)
        let v_new = ln_pflip(&logps, 1, false, rng)[0];
        let logp_out = ftr_logps[v_new];

        // If we chose a singleton view...
        if v_new >= n_views {
            // This will error if v_new is not in the index, and that is a good.
            // thing.
            let tmp_asgn = tmp_asgns.remove(&v_new).unwrap();
            let new_view = view::Builder::from_prior_process(tmp_asgn)
                .seed_from_rng(rng)
                .build();
            self.views.push(new_view);
        }

        // If v_new is >= n_views, it means that we chose the singleton view, so
        // we max the new view index to n_views
        let v_new = v_new.min(n_views);

        self.asgn_mut().reassign(col_ix, v_new);
        self.views[v_new].insert_feature(ftr, rng);
        logp_out
    }

    /// Reassign columns to views using the `FiniteCpu` transition
    pub fn reassign_cols_finite_cpu<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        rng: &mut R,
    ) {
        let n_cols = self.n_cols();

        if n_cols == 1 {
            return;
        }

        let draw_alpha = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewPriorProcessParams);
        self.resample_weights(true, rng);
        self.append_empty_view(draw_alpha, rng);

        let log_weights: Vec<f64> =
            self.weights.iter().map(|w| w.ln()).collect();
        let n_cats = self.asgn().n_cats + 1;

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(n_cols);
        for (i, &v) in self.prior_process.asgn.asgn.iter().enumerate() {
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
                            ftr.asgn_score(view.asgn()) + log_weights[v]
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();

            Matrix::from_raw_parts(values, ftrs.len())
        };

        let new_asgn_vec = crate::massflip::massflip(&logps, rng);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, n_cats, rng);
        self.resample_weights(false, rng);
    }

    pub fn loglike(&self) -> f64 {
        let mut loglike: f64 = 0.0;
        for view in &self.views {
            for ftr in view.ftrs.values() {
                loglike += ftr.score();
            }
        }
        loglike
    }

    #[inline]
    pub fn datum(&self, row_ix: usize, col_ix: usize) -> Datum {
        let view_ix = self.asgn().asgn[col_ix];
        self.views[view_ix].datum(row_ix, col_ix).unwrap()
    }

    pub fn resample_weights<R: Rng>(
        &mut self,
        add_empty_component: bool,
        rng: &mut R,
    ) {
        // FIXME: this only works for Dirichlet!
        let dirvec = self.prior_process.weight_vec(add_empty_component);
        let dir = Dirichlet::new(dirvec).unwrap();
        self.weights = dir.draw(rng)
    }

    pub fn component(&self, row_ix: usize, col_ix: usize) -> Component {
        let view_ix = self.asgn().asgn[col_ix];
        let view = &self.views[view_ix];
        let k = view.asgn().asgn[row_ix];
        view.ftrs[&col_ix].component(k)
    }

    /// Remove the view, but do not adjust any other metadata
    #[inline]
    fn drop_view(&mut self, view_ix: usize) {
        // view goes out of scope and is dropped
        let _view = self.views.remove(view_ix);
    }

    fn append_empty_view<R: Rng>(
        &mut self,
        draw_process_params: bool,
        rng: &mut R,
    ) {
        let asgn_bldr =
            AssignmentBuilder::new(self.n_rows()).with_seed(rng.gen());

        let mut process = self.views[0].prior_process.process.clone();
        if draw_process_params {
            process.reset_params(rng);
        };

        let prior_process = asgn_bldr.with_process(process).build().unwrap();

        let view = view::Builder::from_prior_process(prior_process)
            .seed_from_rng(rng)
            .build();

        self.views.push(view)
    }

    #[inline]
    pub fn impute_bounds(&self, col_ix: usize) -> Option<(f64, f64)> {
        let view_ix = self.asgn().asgn[col_ix];
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

    /// Remove the datum from the table and return it, if it exists
    pub fn remove_datum(
        &mut self,
        row_ix: usize,
        col_ix: usize,
    ) -> Option<Datum> {
        let view_ix = self.asgn().asgn[col_ix];
        self.views[view_ix].remove_datum(row_ix, col_ix)
    }

    pub fn insert_datum(&mut self, row_ix: usize, col_ix: usize, x: Datum) {
        if x.is_missing() {
            self.remove_datum(row_ix, col_ix);
        } else {
            let view_ix = self.asgn().asgn[col_ix];
            self.views[view_ix].insert_datum(row_ix, col_ix, x);
        }
    }

    pub fn drop_data(&mut self) {
        let _data = self.take_data();
    }

    // Delete the top/front n rows.
    pub fn del_rows_at<R: Rng>(&mut self, ix: usize, n: usize, rng: &mut R) {
        self.views
            .iter_mut()
            .for_each(|view| view.del_rows_at(ix, n, rng));
    }

    // Delete a column from the table
    pub fn del_col<R: Rng>(&mut self, ix: usize, rng: &mut R) {
        let zi = self.asgn().asgn[ix];
        let is_singleton = self.asgn().counts[zi] == 1;

        self.asgn_mut().unassign(ix);
        self.asgn_mut().asgn.remove(ix);

        if is_singleton {
            self.views.remove(zi);
        } else {
            self.views[zi].remove_feature(ix);
        }

        // Reindex step
        // self.n_cols counts the number of features in views, so this should be
        // accurate after the remove step above
        for i in ix..self.n_cols() {
            let zi = self.asgn().asgn[i];
            let mut ftr = self.views[zi].remove_feature(i + 1).unwrap();
            ftr.set_id(i);
            self.views[zi].ftrs.insert(ftr.id(), ftr);
        }

        self.resample_weights(false, rng);
    }

    pub fn repop_data(&mut self, mut data: BTreeMap<usize, FeatureData>) {
        if data.len() != self.n_cols() {
            panic!("Data length and state.n_cols differ");
        } else if (0..self.n_cols()).any(|k| !data.contains_key(&k)) {
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
    fn refresh_suffstats<R: Rng>(&mut self, rng: &mut R) {
        self.views.iter_mut().for_each(|v| v.refresh_suffstats(rng));
    }

    pub fn col_weights(&self, col_ix: usize) -> Vec<f64> {
        let view_ix = self.asgn().asgn[col_ix];
        self.views[view_ix].prior_process.weight_vec(false)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::state::Builder;

    use lace_codebook::ColType;

    #[test]
    fn extract_ftr_non_singleton() {
        let mut state = Builder::new()
            .n_rows(50)
            .column_configs(
                4,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_views(2)
            .build()
            .expect("Failed to build state");

        assert_eq!(state.asgn().asgn, vec![0, 0, 1, 1]);

        let ftr = state.extract_ftr(1);

        assert_eq!(state.n_views(), 2);
        assert_eq!(state.views[0].ftrs.len(), 1);
        assert_eq!(state.views[1].ftrs.len(), 2);

        assert_eq!(state.asgn().asgn, vec![0, usize::max_value(), 1, 1]);
        assert_eq!(state.asgn().counts, vec![1, 2]);
        assert_eq!(state.asgn().n_cats, 2);

        assert_eq!(ftr.id(), 1);
    }

    #[test]
    fn extract_ftr_singleton_low() {
        let mut state = Builder::new()
            .n_rows(50)
            .column_configs(
                3,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_views(2)
            .build()
            .expect("Failed to build state");

        assert_eq!(state.asgn().asgn, vec![0, 1, 1]);

        let ftr = state.extract_ftr(0);

        assert_eq!(state.n_views(), 1);
        assert_eq!(state.views[0].ftrs.len(), 2);

        assert_eq!(state.asgn().asgn, vec![usize::max_value(), 0, 0]);
        assert_eq!(state.asgn().counts, vec![2]);
        assert_eq!(state.asgn().n_cats, 1);

        assert_eq!(ftr.id(), 0);
    }

    #[test]
    fn gibbs_col_transition_smoke() {
        let mut rng = rand::thread_rng();
        let mut state = Builder::new()
            .n_rows(50)
            .column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_views(4)
            .n_cats(5)
            .build()
            .expect("Failed to build state");

        let config = StateUpdateConfig {
            n_iters: 100,
            transitions: vec![StateTransition::ColumnAssignment(
                ColAssignAlg::Gibbs,
            )],
        };

        state.update(config, &mut rng);
    }

    #[test]
    fn gibbs_row_transition_smoke() {
        let mut rng = rand::thread_rng();
        let mut state = Builder::new()
            .n_rows(10)
            .column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_views(4)
            .n_cats(5)
            .build()
            .expect("Failed to build state");

        let config = StateUpdateConfig {
            n_iters: 100,
            transitions: vec![StateTransition::RowAssignment(
                RowAssignAlg::Gibbs,
            )],
        };
        state.update(config, &mut rng);
    }

    #[test]
    fn update_should_stop_at_max_iters() {
        let mut rng = rand::thread_rng();

        let n_iters = 37;
        let config = StateUpdateConfig {
            n_iters,
            ..Default::default()
        };

        let colmd = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        let mut state = Builder::new()
            .column_configs(10, colmd)
            .n_rows(1000)
            .build()
            .unwrap();

        state.update(config, &mut rng);

        assert_eq!(state.diagnostics.loglike.len(), n_iters);
    }

    #[test]
    fn flatten_cols() {
        let mut rng = rand::thread_rng();
        let colmd = ColType::Continuous {
            hyper: None,
            prior: None,
        };
        let mut state = Builder::new()
            .column_configs(20, colmd)
            .n_rows(10)
            .n_views(5)
            .build()
            .unwrap();

        assert_eq!(state.n_views(), 5);
        assert_eq!(state.n_cols(), 20);

        state.flatten_cols(&mut rng);
        assert_eq!(state.n_views(), 1);
        assert_eq!(state.n_cols(), 20);

        assert!(state.asgn().asgn.iter().all(|&z| z == 0))
    }
}

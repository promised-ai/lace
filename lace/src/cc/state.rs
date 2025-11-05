mod builder;
use std::collections::BTreeMap;
use std::f64::NEG_INFINITY;

pub use builder::BuildStateError;
pub use builder::Builder;
use rand::seq::SliceRandom as _;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use rv::dist::Dirichlet;
use rv::misc::ln_pflip;
use rv::traits::*;
use serde::Deserialize;
use serde::Serialize;

use crate::cc::alg::ColAssignAlg;
use crate::cc::alg::RowAssignAlg;
use crate::cc::config::StateUpdateConfig;
use crate::cc::feature::geweke::gen_geweke_col_models;
use crate::cc::feature::ColModel;
use crate::cc::feature::Component;
use crate::cc::feature::FType;
use crate::cc::feature::Feature;
use crate::cc::transition::StateTransition;
use crate::cc::view::GewekeViewSummary;
use crate::cc::view::View;
use crate::cc::view::ViewGewekeSettings;
use crate::cc::view::{self};
use crate::consts::geweke_alpha_prior;
use crate::data::Datum;
use crate::data::FeatureData;
use crate::geweke::GewekeModel;
use crate::geweke::GewekeResampleData;
use crate::geweke::GewekeSummarize;
use crate::stats::assignment::Assignment;
use crate::stats::prior_process::Builder as AssignmentBuilder;
use crate::stats::prior_process::PriorProcess;
use crate::stats::prior_process::PriorProcessT;
use crate::stats::prior_process::PriorProcessType;
use crate::stats::prior_process::Process;
use crate::stats::MixtureType;
use crate::utils::unused_components;
use crate::utils::Matrix;

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
            .map(|_| Xoshiro256Plus::from_rng(&mut rng))
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
            .map(|_| Xoshiro256Plus::from_rng(&mut rng))
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
            .map(|_| Xoshiro256Plus::from_rng(&mut rng))
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
    pub fn flatten_cols<R: Rng>(&mut self, mut rng: &mut R) {
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
        use rv::misc::pflip;

        if self.n_views() == 0 {
            self.views.push(view::Builder::new(0).build())
        }

        let k = self.n_views();
        let p = (k as f64).recip();
        ftrs.drain(..).for_each(|mut ftr| {
            ftr.set_id(self.n_cols());
            self.asgn_mut().push_unassigned();
            // insert into random existing view
            let view_ix = pflip(&vec![p; k], None, &mut rng);
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
            AssignmentBuilder::new(self.n_rows()).with_seed(rng.random());
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
            let seeds: Vec<u64> = (0..m).map(|_| rng.random()).collect();
            self.create_tmp_assigns(n_views, update_process_params, &seeds)
        };

        tmp_asgns.iter().for_each(|(_, tmp_asgn)| {
            let singleton_logp = ftr.asgn_score(&tmp_asgn.asgn);
            ftr_logps.push(singleton_logp);
            logps.push(p_singleton + singleton_logp);
        });

        debug_assert_eq!(n_views + m, logps.len());

        // Gibbs step (draw from categorical)
        let v_new = ln_pflip(&logps, false, rng);
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

    #[inline]
    pub fn reassign_col_gibbs<R: Rng>(
        &mut self,
        col_ix: usize,
        update_process_params: bool,
        rng: &mut R,
    ) -> f64 {
        let ftr = self.extract_ftr(col_ix);
        self.insert_feature(ftr, update_process_params, rng)
    }

    /// Reassign all columns using the Gibbs transition.
    ///
    /// # Notes
    /// The transitions are passed to ensure that Geweke tests on subsets of
    /// transitions will still pass. For example, if we are not doing the
    /// `ViewAlpha` transition, we should not draw an alpha from the prior for
    /// the singleton view; instead we should use the existing view alpha.
    pub fn reassign_cols_gibbs<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        rng: &mut R,
    ) {
        if self.n_cols() == 1 {
            return;
        }

        let update_process_params = transitions.contains(&StateTransition::ViewPriorProcessParams);

        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut col_ixs: Vec<usize> = (0..self.n_cols()).collect();
        col_ixs.shuffle(rng);

        col_ixs.drain(..).for_each(|col_ix| {
            self.reassign_col_gibbs(col_ix, update_process_params, rng);
        })
    }

    /// Gibbs column transition where column transition probabilities are pre-
    /// computed in parallel
    pub fn reassign_cols_gibbs_precomputed<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        mut rng: &mut R,
    ) {
        if self.n_cols() == 1 {
            return;
        }

        // Check if we're drawing view alpha. If not, we use the user-specified
        // alpha value for all temporary, singleton assignments
        let draw_process_params = transitions.contains(&StateTransition::ViewPriorProcessParams);

        // determine the number of columns for which to pre-compute transition
        // probabilities
        let batch_size: usize = rayon::current_num_threads() * 2;
        let m: usize = 3;

        // Set the order of the algorithm
        let mut col_ixs: Vec<usize> = (0..self.n_cols()).collect();
        col_ixs.shuffle(rng);

        let n_cols = col_ixs.len();
        // TODO: Can use `unstable_div_ceil` to make this shorter, when it lands
        // in stable. See:
        // https://doc.rust-lang.org/std/primitive.usize.html#:~:text=unchecked_sub-,unstable_div_ceil,-unstable_div_floor
        let n_batches = if n_cols % batch_size == 0 {
            n_cols / batch_size
        } else {
            n_cols / batch_size + 1
        };

        // FIXME: Only works for Dirichlet Process!
        // The partial alpha required for the singleton columns. Since we have
        // `m` singletons to try, we have to divide alpha by m so the singleton
        // proposal as a whole has the correct mass
        let n_views = self.n_views();
        let a_part = self.prior_process.process.ln_singleton_weight(n_views)
            / (m as f64).ln();

        for _ in 0..n_batches {
            // Number of views at the start of the pre-computation
            let end_point = batch_size.min(col_ixs.len());

            // Thread RNGs for parallelism
            let mut t_rngs: Vec<_> = (0..end_point)
                .map(|_| Xoshiro256Plus::from_rng(&mut rng))
                .collect();

            let mut pre_comps = col_ixs
                .par_drain(..end_point)
                .zip(t_rngs.par_drain(..))
                .map(|(col_ix, mut t_rng)| {
                    // let mut logps = vec![0_f64; n_views];

                    let view_ix = self.asgn().asgn[col_ix];
                    let mut logps: Vec<f64> = self
                        .views
                        .iter()
                        .map(|view| {
                            // TODO: we can use Feature::score instead of asgn_score
                            // when the view index is this_view_ix
                            self.feature(col_ix).asgn_score(view.asgn())
                        })
                        .collect();

                    // Always propose new singletons
                    let tmp_asgn_seeds: Vec<u64> =
                        (0..m).map(|_| t_rng.random()).collect();

                    let tmp_asgns = self.create_tmp_assigns(
                        self.n_views(),
                        draw_process_params,
                        &tmp_asgn_seeds,
                    );

                    let ftr = self.feature(col_ix);

                    // TODO: might be faster with an iterator?
                    for asgn in tmp_asgns.values() {
                        logps.push(ftr.asgn_score(&asgn.asgn) + a_part);
                    }

                    (col_ix, view_ix, logps, tmp_asgn_seeds)
                })
                .collect::<Vec<(usize, usize, Vec<f64>, Vec<u64>)>>();

            for _ in 0..pre_comps.len() {
                let (col_ix, this_view_ix, mut logps, seeds) =
                    pre_comps.pop().unwrap();

                let is_singleton = self.asgn().counts[this_view_ix] == 1;

                let n_views = self.n_views();
                logps.iter_mut().take(n_views).enumerate().for_each(
                    |(k, logp)| {
                        // add the CRP component to the log likelihood. We must
                        // remove the contribution to the counts of the current
                        // column.
                        let ct = self.asgn().counts[k] as f64;
                        let ln_ct = if k == this_view_ix {
                            // Note that if ct == 1 this is a singleton in which
                            // case the CRP component will be log(0), which
                            // means this component will never be selected,
                            // which is exactly what we want because columns
                            // must be 'removed' from the table as a part of
                            // gibbs kernel. This simulates that removal.
                            (ct - 1.0).ln()
                        } else {
                            ct.ln()
                        };
                        *logp += ln_ct;
                    },
                );

                let mut v_new = ln_pflip(&logps, false, rng);

                if v_new != this_view_ix {
                    if v_new >= n_views {
                        // Moved to a singleton
                        let seed_ix = v_new - n_views;
                        let seed = seeds[seed_ix];

                        let prior_process =
                            self.create_tmp_assign(draw_process_params, seed);

                        let new_view =
                            view::Builder::from_prior_process(prior_process)
                                .seed_from_rng(&mut rng)
                                .build();

                        self.views.push(new_view);
                        v_new = n_views;

                        // compute likelihood of the rest of the columns under
                        // the new view
                        pre_comps.iter_mut().for_each(
                            |(col_ix, _, ref mut logps, _)| {
                                let logp = self.feature(*col_ix).asgn_score(
                                    self.views.last().unwrap().asgn(),
                                );
                                logps.insert(n_views, logp);
                            },
                        )
                    }

                    if is_singleton {
                        // A singleton was destroyed
                        if v_new >= this_view_ix {
                            // if the view we're assigning to has a greater
                            // index than the one we destroyed, we have to
                            // decrement v_new to maintain order because the
                            // destroyed singleton will be removed in
                            // `extract_ftr`.
                            v_new -= 1;
                        }
                        pre_comps.iter_mut().for_each(|(_, vix, logps, _)| {
                            if this_view_ix < *vix {
                                *vix -= 1;
                            }
                            logps.remove(this_view_ix);
                        })
                    }
                }

                // Unassign, reassign, and insert the feature into the
                // desired view.
                // FIXME: This really shouldn't happen if the assignment doesn't
                // change -- it's extra work for no reason. The reason that this
                // is out here instead of in the above if/else is because for
                // some reason, Engine::insert_data requires the column to be
                // rebuilt...
                let ftr = self.extract_ftr(col_ix);
                self.asgn_mut().reassign(col_ix, v_new);
                self.views[v_new].insert_feature(ftr, rng);
            }
        }
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

        let draw_alpha = transitions.contains(&StateTransition::ViewPriorProcessParams);
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

        let new_asgn_vec = crate::cc::massflip::massflip(&logps, rng);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, n_cats, rng);
        self.resample_weights(false, rng);
    }

    /// Reassign columns to views using the improved slice sampler
    pub fn reassign_cols_slice<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        rng: &mut R,
    ) {
        if self.n_cols() == 1 {
            return;
        }

        self.resample_weights(false, rng);

        let n_cols = self.n_cols();

        let weights: Vec<f64> = {
            let dirvec = self.prior_process.weight_vec_unnormed(true);
            // FIXME: this only works for Dirichlet process!
            let dir = Dirichlet::new(dirvec).unwrap();
            dir.draw(rng)
        };

        let us: Vec<f64> = self
            .asgn()
            .asgn
            .iter()
            .map(|&zi| {
                let wi: f64 = weights[zi];
                let u: f64 = rng.random();
                u * wi
            })
            .collect();

        let u_star: f64 =
            us.iter()
                .fold(1.0, |umin, &ui| if ui < umin { ui } else { umin });

        // Variable shadowing
        let weights = self
            .prior_process
            .process
            .slice_sb_extend(weights, u_star, rng);

        let n_new_views = weights.len() - self.weights.len();
        let n_views = weights.len();

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(n_cols);
        for (i, &v) in self.prior_process.asgn.iter().enumerate() {
            ftrs.push(
                self.views[v].remove_feature(i).expect("Feature missing"),
            );
        }

        let draw_alpha = transitions.contains(&StateTransition::ViewPriorProcessParams);
        for _ in 0..n_new_views {
            self.append_empty_view(draw_alpha, rng);
        }

        // Initialize truncated log probabilities
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
                                ftr.asgn_score(view.asgn())
                            } else {
                                NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();

            Matrix::from_raw_parts(values, ftrs.len())
        };

        let new_asgn_vec =
            crate::cc::massflip::massflip_slice_mat_par(&logps, rng);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, n_views, rng);
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

    fn integrate_finite_asgn<R: Rng>(
        &mut self,
        mut new_asgn_vec: Vec<usize>,
        mut ftrs: Vec<ColModel>,
        n_views: usize,
        rng: &mut R,
    ) {
        let unused_views = unused_components(n_views, &new_asgn_vec);

        for v in unused_views {
            self.drop_view(v);
            for z in new_asgn_vec.iter_mut() {
                if *z > v {
                    *z -= 1
                };
            }
        }

        self.asgn_mut()
            .set_asgn(new_asgn_vec)
            .expect("new_asgn_vec is invalid");

        for (ftr, &v) in ftrs.drain(..).zip(self.prior_process.asgn.asgn.iter())
        {
            self.views[v].insert_feature(ftr, rng)
        }
    }

    /// Extract a feature from its view, unassign it, and drop the view if it
    /// is a singleton.
    fn extract_ftr(&mut self, ix: usize) -> ColModel {
        let v = self.asgn().asgn[ix];
        let ct = self.asgn().counts[v];
        let ftr = self.views[v].remove_feature(ix).unwrap();
        if ct == 1 {
            self.drop_view(v);
        }
        self.asgn_mut().unassign(ix);
        ftr
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
        // View goes out of scope and is dropped
        let _view = self.views.remove(view_ix);
    }

    fn append_empty_view<R: Rng>(
        &mut self,
        draw_process_params: bool,
        rng: &mut R,
    ) {
        let asgn_bldr =
            AssignmentBuilder::new(self.n_rows()).with_seed(rng.random());

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

// Geweke
// ======

#[derive(Clone, Serialize, Deserialize)]
pub struct StateGewekeSettings {
    /// The number of columns/features in the state
    pub n_cols: usize,
    /// The number of rows in the state
    pub n_rows: usize,
    /// Column Model types
    pub cm_types: Vec<FType>,
    /// Which transitions to do
    pub transitions: Vec<StateTransition>,
    /// Which prior process to use for the State assignment
    pub state_process_type: PriorProcessType,
    /// Which prior process to use for the View assignment
    pub view_process_type: PriorProcessType,
}

impl StateGewekeSettings {
    pub fn new(
        n_rows: usize,
        cm_types: Vec<FType>,
        state_process_type: PriorProcessType,
        view_process_type: PriorProcessType,
    ) -> Self {
        use crate::cc::transition::DEFAULT_STATE_TRANSITIONS;

        StateGewekeSettings {
            n_cols: cm_types.len(),
            n_rows,
            cm_types,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
            state_process_type,
            view_process_type,
        }
    }

    pub fn new_dirichlet_process(n_rows: usize, cm_types: Vec<FType>) -> Self {
        use crate::cc::transition::DEFAULT_STATE_TRANSITIONS;

        StateGewekeSettings {
            n_cols: cm_types.len(),
            n_rows,
            cm_types,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        }
    }

    pub fn do_col_asgn_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, StateTransition::ColumnAssignment(_)))
    }

    pub fn do_row_asgn_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, StateTransition::RowAssignment(_)))
    }

    pub fn do_process_params_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, StateTransition::StatePriorProcessParams))
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
            n_rows: 0,
            n_cols: 0,
            cm_types: vec![],
            transitions: s
                .transitions
                .iter()
                .filter_map(|&st| st.try_into().ok())
                .collect(),
            process_type: s.view_process_type,
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
    pub n_views: Option<usize>,
    /// CRP alpha
    pub alpha: Option<f64>,
    /// The summary for each view
    pub views: Vec<GewekeViewSummary>,
}

impl From<&GewekeStateSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeStateSummary) -> Self {
        let mut map: BTreeMap<String, f64> = BTreeMap::new();

        if let Some(n_views) = value.n_views {
            map.insert("n views".into(), n_views as f64);
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
            n_cols: 0,
            n_rows: 0,
            cm_types: vec![],
            transitions: settings
                .transitions
                .iter()
                .filter_map(|&st| st.try_into().ok())
                .collect(),
            process_type: settings.view_process_type,
        };

        GewekeStateSummary {
            n_views: if settings.do_col_asgn_transition() {
                Some(self.asgn().n_cats)
            } else {
                None
            },
            alpha: if settings.do_process_params_transition() {
                Some(match self.prior_process.process {
                    Process::Dirichlet(ref inner) => inner.alpha,
                    Process::PitmanYor(ref inner) => inner.alpha,
                })
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
        use crate::stats::prior_process::Dirichlet as PDirichlet;

        let has_transition = |t: StateTransition, s: &StateGewekeSettings| {
            s.transitions.contains(&t)
        };
        // TODO: Generate new rng from randomly-drawn seed
        // TODO: Draw features properly depending on the transitions
        let do_ftr_prior_transition =
            has_transition(StateTransition::FeaturePriors, settings);

        let do_state_process_transition =
            has_transition(StateTransition::StatePriorProcessParams, settings);

        let do_view_process_transition =
            has_transition(StateTransition::ViewPriorProcessParams, settings);

        let do_col_asgn_transition = settings.do_col_asgn_transition();
        let do_row_asgn_transition = settings.do_row_asgn_transition();

        let mut ftrs = gen_geweke_col_models(
            &settings.cm_types,
            settings.n_rows,
            do_ftr_prior_transition,
            &mut rng,
        );

        let n_cols = ftrs.len();

        let state_prior_process = {
            let process = if do_state_process_transition {
                Process::Dirichlet(PDirichlet::from_prior(
                    geweke_alpha_prior(),
                    &mut rng,
                ))
            } else {
                Process::Dirichlet(PDirichlet {
                    alpha_prior: geweke_alpha_prior(),
                    alpha: 1.0,
                })
            };

            if do_col_asgn_transition {
                AssignmentBuilder::new(n_cols)
            } else {
                AssignmentBuilder::new(n_cols).flat()
            }
            .with_process(process.clone())
            .seed_from_rng(&mut rng)
            .build()
            .unwrap()
        };

        let view_asgn_bldr = if do_row_asgn_transition {
            AssignmentBuilder::new(settings.n_rows)
        } else {
            AssignmentBuilder::new(settings.n_rows).flat()
        };

        let mut views: Vec<View> = (0..state_prior_process.asgn.n_cats)
            .map(|_| {
                // may need to redraw the process params from the prior many
                // times, so Process construction must be a generating function
                let process = if do_view_process_transition {
                    Process::Dirichlet(PDirichlet::from_prior(
                        geweke_alpha_prior(),
                        &mut rng,
                    ))
                } else {
                    Process::Dirichlet(PDirichlet {
                        alpha_prior: geweke_alpha_prior(),
                        alpha: 1.0,
                    })
                };

                let asgn = view_asgn_bldr
                    .clone()
                    .seed_from_rng(&mut rng)
                    .with_process(process.clone())
                    .build()
                    .unwrap();
                view::Builder::from_prior_process(asgn)
                    .seed_from_rng(&mut rng)
                    .build()
            })
            .collect();

        for (&v, ftr) in
            state_prior_process.asgn.asgn.iter().zip(ftrs.drain(..))
        {
            views[v].geweke_init_feature(ftr, &mut rng);
        }

        State {
            views,
            weights: state_prior_process.weight_vec(false),
            prior_process: state_prior_process,
            score: StateScoreComponents::default(),
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
        };

        self.refresh_suffstats(&mut rng);
        self.update(config, &mut rng);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cc::state::Builder;
    use crate::codebook::ColType;

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
        let mut rng = rand::rng();
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
        let mut rng = rand::rng();
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
            let state = State::geweke_from_prior(settings, &mut rng);

            // Column assignment is not flat
            if state.asgn().asgn.iter().any(|&zi| zi != 0) {
                cols_always_flat = false;
            }

            let alpha = match state.prior_process.process {
                Process::Dirichlet(ref inner) => inner.alpha,
                Process::PitmanYor(ref inner) => inner.alpha,
            };

            if !basically_one(alpha) {
                state_alpha_1 = false;
            }

            // 2. Check the column priors
            for view in state.views.iter() {
                // Check the view assignment priors
                // Check the view assignments aren't flat
                if view.asgn().asgn.iter().any(|&zi| zi != 0) {
                    rows_always_flat = false;
                }
                let view_alpha = match view.prior_process.process {
                    Process::Dirichlet(ref inner) => inner.alpha,
                    Process::PitmanYor(ref inner) => inner.alpha,
                };

                if !basically_one(view_alpha) {
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
        let settings = StateGewekeSettings::new_dirichlet_process(
            50,
            vec![FType::Continuous; 40],
        );
        let mut rng = rand::rng();
        let result = test_asgn_flatness(&settings, 10, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_row_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
                StateTransition::StatePriorProcessParams,
                StateTransition::ViewPriorProcessParams,
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_col_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
                StateTransition::StatePriorProcessParams,
                StateTransition::ViewPriorProcessParams,
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_row_or_col_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::StatePriorProcessParams,
                StateTransition::ViewPriorProcessParams,
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(result.rows_always_flat);
        assert!(result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_alpha_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
                StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(result.state_alpha_1);
        assert!(result.view_alphas_1);
    }

    #[test]
    fn update_should_stop_at_max_iters() {
        let mut rng = rand::rng();

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
        let mut rng = rand::rng();
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

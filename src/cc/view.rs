use std::collections::BTreeMap;
use std::f64::NEG_INFINITY;

use braid_flippers::massflip_slice_mat_par;
use braid_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use braid_stats::prior::CrpPrior;
use braid_stats::Datum;
use braid_utils::{unused_components, Matrix};
use rand::{seq::SliceRandom as _, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::Dirichlet;
use rv::misc::ln_pflip;
use rv::traits::Rv;
use serde::{Deserialize, Serialize};

use crate::cc::feature::geweke::{gen_geweke_col_models, ColumnGewekeSettings};
use crate::cc::feature::FeatureData;
use crate::cc::geweke::GewekeColumnSummary;
use crate::cc::transition::ViewTransition;
use crate::cc::{
    Assignment, AssignmentBuilder, ColModel, FType, Feature, RowAssignAlg,
};
use crate::misc::massflip;

/// A cross-categorization view of columns/features
///
/// View is a multivariate generalization of the standard Diriclet-process
/// mixture model (DPGMM). `View` captures a joint distribution over its
/// columns by assuming the columns are dependent.
#[derive(Serialize, Deserialize, Clone)]
pub struct View {
    /// A Map of features indexed by the feature ID
    pub ftrs: BTreeMap<usize, ColModel>,
    /// The assignment of rows to categories
    pub asgn: Assignment,
    /// The weights of each category
    pub weights: Vec<f64>,
}

/// Builds a `View`
pub struct ViewBuilder {
    nrows: usize,
    alpha_prior: Option<CrpPrior>,
    asgn: Option<Assignment>,
    ftrs: Option<Vec<ColModel>>,
    seed: Option<u64>,
}

impl ViewBuilder {
    /// Start building a view with a given number of rows
    pub fn new(nrows: usize) -> Self {
        ViewBuilder {
            nrows,
            asgn: None,
            alpha_prior: None,
            ftrs: None,
            seed: None,
        }
    }

    /// Start building a view with a given row assignment.
    ///
    /// Note that the number of rows will be the assignment length.
    pub fn from_assignment(asgn: Assignment) -> Self {
        ViewBuilder {
            nrows: asgn.len(),
            asgn: Some(asgn),
            alpha_prior: None, // is ignored in asgn set
            ftrs: None,
            seed: None,
        }
    }

    /// Put a custom `Gamma` prior on the CRP alpha
    pub fn with_alpha_prior(mut self, alpha_prior: CrpPrior) -> Self {
        if self.asgn.is_some() {
            panic!("Cannot add alpha_prior once Assignment added");
        } else {
            self.alpha_prior = Some(alpha_prior);
            self
        }
    }

    /// Add features to the `View`
    pub fn with_features(mut self, ftrs: Vec<ColModel>) -> Self {
        self.ftrs = Some(ftrs);
        self
    }

    /// Set the RNG seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the RNG seed from another RNG
    pub fn seed_from_rng<R: Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// Build the `View` and consume the builder
    pub fn build(self) -> View {
        let mut rng = match self.seed {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_entropy(),
        };

        let asgn = match self.asgn {
            Some(asgn) => asgn,
            None => {
                if self.alpha_prior.is_none() {
                    AssignmentBuilder::new(self.nrows)
                        .seed_from_rng(&mut rng)
                        .build()
                        .unwrap()
                } else {
                    AssignmentBuilder::new(self.nrows)
                        .with_prior(self.alpha_prior.unwrap())
                        .seed_from_rng(&mut rng)
                        .build()
                        .unwrap()
                }
            }
        };

        let weights = asgn.weights();
        let mut ftr_tree = BTreeMap::new();
        if let Some(mut ftrs) = self.ftrs {
            for mut ftr in ftrs.drain(..) {
                ftr.reassign(&asgn, &mut rng);
                ftr_tree.insert(ftr.id(), ftr);
            }
        }

        View {
            ftrs: ftr_tree,
            asgn,
            weights,
        }
    }
}

unsafe impl Send for View {}
unsafe impl Sync for View {}

impl View {
    /// The number of rows in the `View`
    #[inline]
    pub fn nrows(&self) -> usize {
        self.asgn.asgn.len()
    }

    /// The number of columns in the `View`
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ftrs.len()
    }

    /// The number of columns/features
    #[inline]
    pub fn len(&self) -> usize {
        self.ncols()
    }

    /// returns true if there are no features
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ncols() == 0
    }

    /// The number of categories
    #[inline]
    pub fn ncats(&self) -> usize {
        self.asgn.ncats
    }

    /// The current value of the CPR alpha parameter
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.asgn.alpha
    }

    // Extend the columns by a number of cells, increasing the total number of
    // rows. The added entries will be empty.
    pub fn extend_cols(&mut self, nrows: usize) {
        (0..nrows).for_each(|_| self.asgn.push_unassigned());
        self.ftrs.values_mut().for_each(|ftr| {
            (0..nrows).for_each(|_| ftr.append_datum(Datum::Missing))
        })
    }

    pub fn insert_datum(&mut self, row_ix: usize, col_ix: usize, x: Datum) {
        let k = self.asgn.asgn[row_ix];
        let is_assigned = k != usize::max_value();

        let ftr = self.ftrs.get_mut(&col_ix).unwrap();

        if is_assigned {
            ftr.forget_datum(row_ix, k);
            ftr.insert_datum(row_ix, x);
            ftr.observe_datum(row_ix, k);
        } else {
            ftr.insert_datum(row_ix, x);
        }
    }

    // pub fn append_rows(
    //     &mut self,
    //     new_rows: Vec<&AppendRowsData>,
    //     mut rng: &mut impl Rng,
    // ) {
    //     assert_eq!(self.ncols(), new_rows.len());

    //     let nrows = self.nrows();
    //     let n_new_rows = new_rows[0].len();
    //     for row_ix in 0..n_new_rows {
    //         self.asgn.push_unassigned();
    //         for ftr_rows in new_rows.iter() {
    //             self.ftrs
    //                 .get_mut(&ftr_rows.col_ix)
    //                 .unwrap()
    //                 .append_datum(ftr_rows.data[row_ix].clone())
    //         }
    //         // Insert row by Gibbs
    //         self.reinsert_row(nrows + row_ix, &mut rng);
    //     }
    // }

    /// Get the log PDF/PMF of the datum at `row_ix` in feature `col_ix`
    #[inline]
    pub fn logp_at(&self, row_ix: usize, col_ix: usize) -> Option<f64> {
        let k = self.asgn.asgn[row_ix];
        self.ftrs[&col_ix].logp_at(row_ix, k)
    }

    /// The probability of the row at `row_ix` belonging to cluster `k` given
    /// the data already assigned to category `k` with all component parameters
    /// marginalized away
    #[inline]
    pub fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        self.ftrs
            .values()
            .fold(0.0, |acc, ftr| acc + ftr.predictive_score_at(row_ix, k))
    }

    /// The marginal likelihood of `row_ix`
    #[inline]
    pub fn singleton_score(&self, row_ix: usize) -> f64 {
        self.ftrs
            .values()
            .fold(0.0, |acc, ftr| acc + ftr.singleton_score(row_ix))
    }

    /// get the datum at `row_ix` under the feature with id `col_ix`
    #[inline]
    pub fn datum(&self, row_ix: usize, col_ix: usize) -> Option<Datum> {
        if self.ftrs.contains_key(&col_ix) {
            Some(self.ftrs[&col_ix].datum(row_ix))
        } else {
            None
        }
    }

    /// Perform MCMC transitions on the view
    pub fn step(
        &mut self,
        row_asgn_alg: RowAssignAlg,
        transitions: &[ViewTransition],
        mut rng: &mut impl Rng,
    ) {
        for transition in transitions {
            match transition {
                ViewTransition::Alpha => self.update_alpha(&mut rng),
                ViewTransition::RowAssignment => {
                    self.reassign(row_asgn_alg, &mut rng)
                }
                ViewTransition::FeaturePriors => {
                    self.update_prior_params(&mut rng)
                }
                ViewTransition::ComponentParams => {
                    self.update_component_params(&mut rng)
                }
            }
        }
    }

    /// The default MCMC transitions
    pub fn default_transitions() -> Vec<ViewTransition> {
        vec![
            ViewTransition::RowAssignment,
            ViewTransition::Alpha,
            ViewTransition::FeaturePriors,
        ]
    }

    /// Update the state of the `View` by running the `View` MCMC transitions
    /// `n_iter` times.
    #[inline]
    pub fn update(
        &mut self,
        n_iters: usize,
        alg: RowAssignAlg,
        transitions: &[ViewTransition],
        mut rng: &mut impl Rng,
    ) {
        (0..n_iters).for_each(|_| self.step(alg, &transitions, &mut rng))
    }

    /// Update the prior parameters on each feature
    #[inline]
    pub fn update_prior_params(&mut self, mut rng: &mut impl Rng) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.update_prior_params(&mut rng));
    }

    /// Update the component parameters in each feature
    #[inline]
    pub fn update_component_params(&mut self, mut rng: &mut impl Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.update_components(&mut rng);
        }
    }

    /// Reassign the rows to categories
    pub fn reassign(&mut self, alg: RowAssignAlg, mut rng: &mut impl Rng) {
        match alg {
            RowAssignAlg::FiniteCpu => self.reassign_rows_finite_cpu(&mut rng),
            RowAssignAlg::Slice => self.reassign_rows_slice(&mut rng),
            RowAssignAlg::Gibbs => self.reassign_rows_gibbs(&mut rng),
            RowAssignAlg::Sams => self.reassign_rows_sams(&mut rng),
        }
    }

    /// Find all unassigned rows and reassign them using Gibbs
    pub(crate) fn assign_unassigned<R: Rng>(&mut self, mut rng: &mut R) {
        // TODO: Probably some optimization we could do here to no clone. The
        // problem is that I can't iterate on self.asgn then call
        // self.reinsert_row inside the for_each closure
        let mut unassigned_rows: Vec<usize> = self
            .asgn
            .iter()
            .enumerate()
            .filter_map(|(row_ix, &z)| {
                if z == usize::max_value() {
                    Some(row_ix)
                } else {
                    None
                }
            })
            .collect();

        unassigned_rows.drain(..).for_each(|row_ix| {
            self.reinsert_row(row_ix, &mut rng);
        })
    }

    #[inline]
    fn remove_row(&mut self, row_ix: usize) {
        let k = self.asgn.asgn[row_ix];
        let is_singleton = self.asgn.counts[k] == 1;
        self.forget_row(row_ix, k);
        self.asgn.unassign(row_ix);

        if is_singleton {
            self.drop_component(k);
        }
    }

    #[inline]
    fn reinsert_row(&mut self, row_ix: usize, mut rng: &mut impl Rng) {
        let k_new = if self.asgn.ncats == 0 {
            debug_assert!(self.ftrs.values().all(|f| f.k() == 0));
            self.append_empty_component(&mut rng);
            0
        } else {
            let mut logps = self.asgn.log_dirvec(true);
            (0..self.asgn.ncats).for_each(|k| {
                logps[k] += self.predictive_score_at(row_ix, k);
            });
            logps[self.asgn.ncats] += self.singleton_score(row_ix);

            let k_new = ln_pflip(&logps, 1, false, &mut rng)[0];
            if k_new == self.asgn.ncats {
                self.append_empty_component(&mut rng);
            }
            k_new
        };

        self.observe_row(row_ix, k_new);
        self.asgn.reassign(row_ix, k_new);
    }

    /// Use the standard Gibbs kernel to reassign the rows
    #[inline]
    pub fn reassign_rows_gibbs(&mut self, mut rng: &mut impl Rng) {
        let nrows = self.nrows();

        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut row_ixs: Vec<usize> = (0..nrows).map(|i| i).collect();
        row_ixs.shuffle(&mut rng);

        for row_ix in row_ixs {
            self.remove_row(row_ix);
            self.reinsert_row(row_ix, &mut rng);
        }

        // NOTE: The oracle functions use the weights to compute probabilities.
        // Since the Gibbs algorithm uses implicit weights from the partition,
        // it does not explicitly update the weights. Non-updated weights means
        // wrong probabilities. To avoid this, we set the weights by the
        // partition here.
        self.weights = self.asgn.weights();
    }

    /// Use the finite approximation (on the CPU) to reassign the rows
    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut impl Rng) {
        let ncats = self.asgn.ncats;
        let nrows = self.nrows();

        self.resample_weights(true, &mut rng);
        self.append_empty_component(&mut rng);

        // initialize log probabilities
        let ln_weights: Vec<f64> =
            self.weights.iter().map(|&w| w.ln()).collect();
        let logps = Matrix::vtile(ln_weights, nrows);

        self.accum_score_and_integrate_asgn(
            logps,
            ncats + 1,
            RowAssignAlg::FiniteCpu,
            &mut rng,
        );
    }

    /// Use the improved slice algorithm to reassign the rows
    pub fn reassign_rows_slice(&mut self, mut rng: &mut impl Rng) {
        use crate::dist::stick_breaking::sb_slice_extend;
        self.resample_weights(false, &mut rng);

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
                let u: f64 = rng.gen::<f64>();
                u * wi
            })
            .collect();

        let u_star: f64 =
            us.iter()
                .fold(1.0, |umin, &ui| if ui < umin { ui } else { umin });

        let weights =
            sb_slice_extend(weights, self.asgn.alpha, u_star, &mut rng)
                .unwrap();

        let n_new_cats = weights.len() - self.weights.len();
        let ncats = weights.len();

        for _ in 0..n_new_cats {
            self.append_empty_component(&mut rng);
        }

        // initialize truncated log probabilities
        let logps = {
            let mut values = Vec::with_capacity(weights.len() * self.nrows());
            weights.iter().for_each(|w| {
                us.iter().for_each(|ui| {
                    let value = if w >= ui { 0.0 } else { NEG_INFINITY };
                    values.push(value);
                });
            });
            let matrix = Matrix::from_raw_parts(values, ncats);
            debug_assert_eq!(matrix.ncols(), us.len());
            debug_assert_eq!(matrix.nrows(), weights.len());
            matrix
        };

        self.accum_score_and_integrate_asgn(
            logps,
            ncats,
            RowAssignAlg::Slice,
            &mut rng,
        );
    }

    fn accum_score_and_integrate_asgn(
        &mut self,
        mut logps: Matrix<f64>,
        ncats: usize,
        row_alg: RowAssignAlg,
        mut rng: &mut impl Rng,
    ) {
        // TODO: parallelize over rows_mut somehow?
        logps.rows_mut().enumerate().for_each(|(k, mut logp)| {
            self.ftrs.values().for_each(|ftr| {
                ftr.accum_score(&mut logp, k);
            })
        });

        // Implicit transpose does not change the memory layout, just the
        // indexing.
        logps.implicit_transpose();
        debug_assert_eq!(logps.nrows(), self.nrows());

        let new_asgn_vec = match row_alg {
            RowAssignAlg::Slice => massflip_slice_mat_par(&logps, &mut rng),
            _ => massflip(&logps, &mut rng),
        };

        self.integrate_finite_asgn(new_asgn_vec, ncats, &mut rng);
    }

    /// Resample the component weights
    ///
    /// # Note
    ///
    /// Used only for the FinteCpu and Slice algorithms
    pub fn resample_weights(
        &mut self,
        add_empty_component: bool,
        mut rng: &mut impl Rng,
    ) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec).unwrap();
        self.weights = dir.draw(&mut rng)
    }

    pub fn reassign_rows_sams(&mut self, _rng: &mut impl Rng) {
        // Naive, SIS split-merge
        // ======================
        //
        // 1. choose two columns, i and j
        // 2. If z_i == z_j, split(z_i, z_j) else merge(z_i, z_j)
        //        let (i, j) = choose2ixs(self.nrows(), &mut rng);
        //        let zi = self.asgn.asgn[i];
        //        let zj = self.asgn.asgn[j];
        //
        //        if zi == zj {
        //            self.sams_split(i, j, &mut rng);
        //        } else {
        //            self.sams_merge(i, j, &mut rng);
        //        }
        unimplemented!()
    }

    //    fn sams_split(&mut self, i: usize, j: usize, mut rng: impl Rng) {
    //        // Split
    //        // -----
    //        // Def. k := the component to which i and j are currently assigned
    //        // Def. x_k := all the data assigned to component k
    //        // 1. Create a component with x_k
    //        // 2. Create two components: one with the datum at i and one with the
    //        //    datum at j.
    //        // 3. Assign the remaning data to components i or j via SIS
    //        // 4. Do Proposal
    //
    //        // append two empty components
    //        self.append_empty_component(&mut rng);
    //        self.append_empty_component(&mut rng);
    //
    //        let zij = self.asgn.asgn[i]; // The original category
    //        let zi = self.asgn.ncats; // The proposed new category of i
    //        let zj = zi + 1; // The proposed new category of j
    //    }

    //    fn sams_merge(&self, _i: usize, _j: usize, _rng: impl Rng) {
    //        // Merge
    //        // ----
    //        // 1. Create a component with x_i and x_j combined
    //        // 2. Create two components: one with with datum i and one with datum j
    //        // 3. Compute the reverse probability of the given assignment of a
    //        //    split
    //        // 4. Compute the MH acceptance
    //    }

    /// MCMC update on the CPR alpha parameter
    #[inline]
    pub fn update_alpha(&mut self, mut rng: &mut impl Rng) {
        self.asgn
            .update_alpha(braid_consts::MH_PRIOR_ITERS, &mut rng);
    }

    #[inline]
    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.append_empty_component(&mut rng);
        }
    }

    #[inline]
    fn drop_component(&mut self, k: usize) {
        for ftr in self.ftrs.values_mut() {
            ftr.drop_component(k);
        }
    }

    // Cleanup functions
    fn integrate_finite_asgn(
        &mut self,
        mut new_asgn_vec: Vec<usize>,
        ncats: usize,
        mut rng: &mut impl Rng,
    ) {
        // Returns the unused category indices in descending order so that
        // removing the unused components and reindexing requires less
        // bookkeeping
        let unused_cats = unused_components(ncats, &new_asgn_vec);

        for k in unused_cats {
            self.drop_component(k);
            for z in new_asgn_vec.iter_mut() {
                if *z > k {
                    *z -= 1
                };
            }
        }

        self.asgn
            .set_asgn(new_asgn_vec)
            .expect("new asgn is invalid");
        self.resample_weights(false, &mut rng);
        for ftr in self.ftrs.values_mut() {
            ftr.reassign(&self.asgn, &mut rng)
        }
    }

    /// Insert a new `Feature` into the `View`, but draw the feature
    /// components from the prior
    #[inline]
    pub fn init_feature(&mut self, mut ftr: ColModel, mut rng: &mut impl Rng) {
        let id = ftr.id();
        if self.ftrs.contains_key(&id) {
            panic!("Feature {} already in view", id);
        }
        ftr.init_components(self.asgn.ncats, &mut rng);
        ftr.reassign(&self.asgn, &mut rng);
        self.ftrs.insert(id, ftr);
    }

    /// Insert a new `Feature` into the `View`
    #[inline]
    pub fn insert_feature(
        &mut self,
        mut ftr: ColModel,
        mut rng: &mut impl Rng,
    ) {
        let id = ftr.id();
        if self.ftrs.contains_key(&id) {
            panic!("Feature {} already in view", id);
        }
        ftr.reassign(&self.asgn, &mut rng);

        self.ftrs.insert(id, ftr);
    }

    /// Remove and return the `Feature` with `id`. Returns `None` if the `id`
    /// is not found.
    #[inline]
    pub fn remove_feature(&mut self, id: usize) -> Option<ColModel> {
        self.ftrs.remove(&id)
    }

    /// Remove all of the data from the features
    pub fn take_data(&mut self) -> BTreeMap<usize, FeatureData> {
        let mut data: BTreeMap<usize, FeatureData> = BTreeMap::new();
        self.ftrs.iter_mut().for_each(|(id, ftr)| {
            data.insert(*id, ftr.take_data());
        });
        data
    }

    /// Show the data in `row_ix` to the components `k`
    #[inline]
    fn observe_row(&mut self, row_ix: usize, k: usize) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.observe_datum(row_ix, k));
    }

    /// Have the components `k` forgets the data in `row_ix`
    #[inline]
    fn forget_row(&mut self, row_ix: usize, k: usize) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.forget_datum(row_ix, k));
    }

    /// Recompute the sufficient statistics in each component
    #[inline]
    pub fn refresh_suffstats(&mut self, mut rng: &mut impl Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.reassign(&self.asgn, &mut rng);
        }
    }

    /// Get the likelihood of the data in this view given the current assignment
    #[inline]
    pub fn score(&self) -> f64 {
        self.ftrs.values().fold(0.0, |acc, ftr| acc + ftr.score())
    }
}

// Geweke
// ======
/// Configuration of the Geweke test on Views
pub struct ViewGewekeSettings {
    /// The number of columns/features in the view
    pub ncols: usize,
    /// The number of rows in the view
    pub nrows: usize,
    /// The row reassignment algorithm
    pub row_alg: RowAssignAlg,
    /// Column model types
    pub cm_types: Vec<FType>,
    /// Which transitions to run
    pub transitions: Vec<ViewTransition>,
}

impl ViewGewekeSettings {
    pub fn new(nrows: usize, cm_types: Vec<FType>) -> Self {
        ViewGewekeSettings {
            nrows,
            ncols: cm_types.len(),
            row_alg: RowAssignAlg::FiniteCpu,
            cm_types,
            transitions: View::default_transitions(),
        }
    }
}

impl GewekeModel for View {
    fn geweke_from_prior(
        settings: &ViewGewekeSettings,
        mut rng: &mut impl Rng,
    ) -> View {
        let do_ftr_prior_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::FeaturePriors);

        let do_row_asgn_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::RowAssignment);

        let ftrs = gen_geweke_col_models(
            &settings.cm_types,
            settings.nrows,
            do_ftr_prior_transition,
            &mut rng,
        );

        if do_row_asgn_transition {
            ViewBuilder::new(settings.nrows).with_features(ftrs)
        } else {
            let asgn = AssignmentBuilder::new(settings.nrows)
                .flat()
                .seed_from_rng(&mut rng)
                .build()
                .unwrap();
            ViewBuilder::from_assignment(asgn).with_features(ftrs)
        }
        .seed_from_rng(&mut rng)
        .build()
    }

    fn geweke_step(
        &mut self,
        settings: &ViewGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        self.step(settings.row_alg, &settings.transitions, &mut rng);
    }
}

impl GewekeResampleData for View {
    type Settings = ViewGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&ViewGewekeSettings>,
        rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        let col_settings =
            ColumnGewekeSettings::new(self.asgn.clone(), s.transitions.clone());
        for ftr in self.ftrs.values_mut() {
            ftr.geweke_resample_data(Some(&col_settings), rng);
        }
    }
}

/// The View summary for Geweke
#[derive(Clone, Debug)]
pub struct GewekeViewSummary {
    /// The number of categories
    pub ncats: Option<usize>,
    /// The CRP alpha
    pub alpha: Option<f64>,
    /// The summary for each column/feature.
    pub cols: Vec<(usize, GewekeColumnSummary)>,
}

impl From<&GewekeViewSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeViewSummary) -> BTreeMap<String, f64> {
        let mut map: BTreeMap<String, f64> = BTreeMap::new();
        if let Some(ncats) = value.ncats {
            map.insert("ncats".into(), ncats as f64);
        }

        if let Some(alpha) = value.alpha {
            map.insert("crp alpha".into(), alpha);
        }

        value.cols.iter().for_each(|(id, col_summary)| {
            let summary_map: BTreeMap<String, f64> = col_summary.into();
            summary_map.iter().for_each(|(key, value)| {
                let new_key = format!("Col {} {}", id, key);
                map.insert(new_key, *value);
            });
        });
        map
    }
}

impl From<GewekeViewSummary> for BTreeMap<String, f64> {
    fn from(value: GewekeViewSummary) -> BTreeMap<String, f64> {
        Self::from(&value)
    }
}

impl GewekeSummarize for View {
    type Summary = GewekeViewSummary;

    fn geweke_summarize(&self, settings: &ViewGewekeSettings) -> Self::Summary {
        let do_row_asgn_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::RowAssignment);

        let do_alpha_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::Alpha);

        let col_settings = ColumnGewekeSettings::new(
            self.asgn.clone(),
            settings.transitions.clone(),
        );

        GewekeViewSummary {
            ncats: if do_row_asgn_transition {
                Some(self.ncats())
            } else {
                None
            },
            alpha: if do_alpha_transition {
                Some(self.asgn.alpha)
            } else {
                None
            },
            cols: self
                .ftrs
                .values()
                .map(|ftr| (ftr.id(), ftr.geweke_summarize(&col_settings)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cc::{Column, ConjugateComponent, DataContainer};
    use braid_stats::prior::{Ng, NigHyper};
    use rv::dist::Gaussian;

    fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
        let gauss = Gaussian::new(0.0, 1.0).unwrap();
        let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
        let data = DataContainer::new(data_vec);
        let hyper = NigHyper::default();
        let prior = Ng::new(0.0, 1.0, 1.0, 1.0, hyper);

        let ftr = Column::new(id, data, prior);
        ColModel::Continuous(ftr)
    }

    fn gen_gauss_view<R: Rng>(n: usize, mut rng: &mut R) -> View {
        let mut ftrs: Vec<ColModel> = vec![];
        ftrs.push(gen_col(0, n, &mut rng));
        ftrs.push(gen_col(1, n, &mut rng));
        ftrs.push(gen_col(2, n, &mut rng));
        ftrs.push(gen_col(3, n, &mut rng));

        ViewBuilder::new(n)
            .with_features(ftrs)
            .seed_from_rng(&mut rng)
            .build()
    }

    fn extract_components(
        view: &View,
    ) -> Vec<Vec<ConjugateComponent<f64, Gaussian>>> {
        view.ftrs
            .iter()
            .map(|(_, ftr)| {
                if let ColModel::Continuous(f) = ftr {
                    f.components.clone()
                } else {
                    panic!("not a gaussian feature")
                }
            })
            .collect()
    }

    #[test]
    fn extend_cols_adds_empty_unassigned_rows() {
        let mut rng = rand::thread_rng();
        let mut view = gen_gauss_view(10, &mut rng);

        let components_start = extract_components(&view);

        view.extend_cols(2);

        assert_eq!(view.asgn.asgn.len(), 12);
        assert_eq!(view.asgn.asgn[10], usize::max_value());
        assert_eq!(view.asgn.asgn[11], usize::max_value());

        for ftr in view.ftrs.values() {
            assert_eq!(ftr.len(), 12);
        }

        let components_end = extract_components(&view);

        assert_eq!(components_start, components_end);
    }

    #[test]
    fn insert_datum_into_existing_spot_updates_suffstats() {
        let mut rng = rand::thread_rng();
        let mut view = gen_gauss_view(10, &mut rng);

        let components_start = extract_components(&view);

        let view_ix_start = view.asgn.asgn[2];
        let component_start = components_start[3][view_ix_start].clone();

        view.insert_datum(2, 3, Datum::Continuous(20.22));

        let components_end = extract_components(&view);
        let view_ix_end = view.asgn.asgn[2];
        let component_end = components_end[3][view_ix_end].clone();

        assert_ne!(components_start, components_end);
        assert_ne!(component_start, components_end[3][view_ix_start]);
        assert_ne!(component_start, component_end);
    }

    #[test]
    fn insert_datum_into_unassigned_spot_does_not_update_suffstats() {
        let mut rng = rand::thread_rng();
        let mut view = gen_gauss_view(10, &mut rng);

        let components_start = extract_components(&view);

        view.extend_cols(1);

        view.insert_datum(10, 3, Datum::Continuous(20.22));

        let components_end = extract_components(&view);

        assert_eq!(components_start, components_end);
    }
}

use std::collections::BTreeMap;
use std::f64::NEG_INFINITY;

use braid_data::{Datum, FeatureData};
use braid_flippers::massflip_slice_mat_par;
use braid_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::rv::dist::Dirichlet;
use braid_stats::rv::misc::ln_pflip;
use braid_stats::rv::traits::Rv;
use braid_utils::{logaddexp, unused_components, Matrix, Shape};
use rand::{seq::SliceRandom as _, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};

// use crate::cc::feature::geweke::{gen_geweke_col_models, ColumnGewekeSettings};
use crate::alg::RowAssignAlg;
use crate::assignment::{Assignment, AssignmentBuilder};
use crate::feature::geweke::GewekeColumnSummary;
use crate::feature::geweke::{gen_geweke_col_models, ColumnGewekeSettings};
use crate::feature::{ColModel, FType, Feature};
use crate::misc::massflip;
use crate::transition::ViewTransition;

/// A cross-categorization view of columns/features
///
/// View is a multivariate generalization of the standard Diriclet-process
/// mixture model (DPGMM). `View` captures a joint distribution over its
/// columns by assuming the columns are dependent.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct View {
    /// A Map of features indexed by the feature ID
    pub ftrs: BTreeMap<usize, ColModel>,
    /// The assignment of rows to categories
    pub asgn: Assignment,
    /// The weights of each category
    pub weights: Vec<f64>,
}

/// Builds a `View`
pub struct Builder {
    n_rows: usize,
    alpha_prior: Option<CrpPrior>,
    asgn: Option<Assignment>,
    ftrs: Option<Vec<ColModel>>,
    seed: Option<u64>,
}

impl Builder {
    /// Start building a view with a given number of rows
    pub fn new(n_rows: usize) -> Self {
        Builder {
            n_rows,
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
        Builder {
            n_rows: asgn.len(),
            asgn: Some(asgn),
            alpha_prior: None, // is ignored in asgn set
            ftrs: None,
            seed: None,
        }
    }

    /// Put a custom `Gamma` prior on the CRP alpha
    #[must_use]
    pub fn alpha_prior(mut self, alpha_prior: CrpPrior) -> Self {
        if self.asgn.is_some() {
            panic!("Cannot add alpha_prior once Assignment added");
        } else {
            self.alpha_prior = Some(alpha_prior);
            self
        }
    }

    /// Add features to the `View`
    #[must_use]
    pub fn features(mut self, ftrs: Vec<ColModel>) -> Self {
        self.ftrs = Some(ftrs);
        self
    }

    /// Set the RNG seed
    #[must_use]
    pub fn seed_from_u64(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the RNG seed from another RNG
    #[must_use]
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
                    AssignmentBuilder::new(self.n_rows)
                        .seed_from_rng(&mut rng)
                        .build()
                        .unwrap()
                } else {
                    AssignmentBuilder::new(self.n_rows)
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

impl View {
    /// The number of rows in the `View`
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.asgn.asgn.len()
    }

    /// The number of columns in the `View`
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.ftrs.len()
    }

    /// The number of columns/features
    #[inline]
    pub fn len(&self) -> usize {
        self.n_cols()
    }

    /// returns true if there are no features
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_cols() == 0
    }

    /// The number of categories
    #[inline]
    pub fn n_cats(&self) -> usize {
        self.asgn.n_cats
    }

    /// The current value of the CPR alpha parameter
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.asgn.alpha
    }

    // Extend the columns by a number of cells, increasing the total number of
    // rows. The added entries will be empty.
    pub fn extend_cols(&mut self, n_rows: usize) {
        (0..n_rows).for_each(|_| self.asgn.push_unassigned());
        self.ftrs.values_mut().for_each(|ftr| {
            (0..n_rows).for_each(|_| ftr.append_datum(Datum::Missing))
        })
    }

    /// Remove the datum (set as missing) and return it if it existed
    pub fn remove_datum(
        &mut self,
        row_ix: usize,
        col_ix: usize,
    ) -> Option<Datum> {
        let k = self.asgn.asgn[row_ix];
        let is_assigned = k != usize::max_value();

        if is_assigned {
            let ftr = self.ftrs.get_mut(&col_ix).unwrap();
            ftr.take_datum(row_ix, k)
        } else {
            None
        }
    }

    pub fn insert_datum(&mut self, row_ix: usize, col_ix: usize, x: Datum) {
        if x.is_missing() {
            self.remove_datum(row_ix, col_ix);
            return;
        }

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

    /// The probability of the row at `row_ix` belonging to cluster `k` given
    /// the data already assigned to category `k` with all component parameters
    /// marginalized away
    #[inline]
    pub fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        self.ftrs
            .values()
            .fold(0.0, |acc, ftr| acc + ftr.predictive_score_at(row_ix, k))
    }

    #[inline]
    pub fn logm(&self, k: usize) -> f64 {
        self.ftrs.values().map(|ftr| ftr.logm(k)).sum()
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
        transitions: &[ViewTransition],
        mut rng: &mut impl Rng,
    ) {
        for transition in transitions {
            match transition {
                ViewTransition::Alpha => {
                    self.update_alpha(&mut rng);
                }
                ViewTransition::RowAssignment(alg) => {
                    self.reassign(*alg, &mut rng);
                }
                ViewTransition::FeaturePriors => {
                    self.update_prior_params(&mut rng);
                }
                ViewTransition::ComponentParams => {
                    self.update_component_params(&mut rng);
                }
            }
        }
    }

    /// The default MCMC transitions
    pub fn default_transitions() -> Vec<ViewTransition> {
        vec![
            ViewTransition::RowAssignment(RowAssignAlg::FiniteCpu),
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
        transitions: &[ViewTransition],
        mut rng: &mut impl Rng,
    ) {
        (0..n_iters).for_each(|_| self.step(transitions, &mut rng))
    }

    /// Update the prior parameters on each feature
    #[inline]
    pub fn update_prior_params(&mut self, mut rng: &mut impl Rng) -> f64 {
        self.ftrs
            .values_mut()
            .map(|ftr| ftr.update_prior_params(&mut rng))
            .sum()
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
        // Reassignment doesn't make any sense if there is only one row, because
        // there can only be one on component.
        if self.n_rows() < 2 {
            return;
        }
        match alg {
            RowAssignAlg::FiniteCpu => self.reassign_rows_finite_cpu(&mut rng),
            RowAssignAlg::Slice => self.reassign_rows_slice(&mut rng),
            RowAssignAlg::Gibbs => self.reassign_rows_gibbs(&mut rng),
            RowAssignAlg::Sams => self.reassign_rows_sams(&mut rng),
        }
    }

    #[inline]
    pub fn reassign_row_gibbs(
        &mut self,
        row_ix: usize,
        mut rng: &mut impl Rng,
    ) {
        self.remove_row(row_ix);
        self.reinsert_row(row_ix, &mut rng);
    }

    /// Use the standard Gibbs kernel to reassign the rows
    #[inline]
    pub fn reassign_rows_gibbs(&mut self, mut rng: &mut impl Rng) {
        let n_rows = self.n_rows();

        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut row_ixs: Vec<usize> = (0..n_rows).collect();
        row_ixs.shuffle(&mut rng);

        for row_ix in row_ixs {
            self.reassign_row_gibbs(row_ix, &mut rng);
        }

        // NOTE: The oracle functions use the weights to compute probabilities.
        // Since the Gibbs algorithm uses implicit weights from the partition,
        // it does not explicitly update the weights. Non-updated weights means
        // wrong probabilities. To avoid this, we set the weights by the
        // partition here.
        self.weights = self.asgn.weights();
        debug_assert!(self.asgn.validate().is_valid());
    }

    /// Use the finite approximation (on the CPU) to reassign the rows
    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut impl Rng) {
        let n_cats = self.asgn.n_cats;
        let n_rows = self.n_rows();

        self.resample_weights(true, &mut rng);
        self.append_empty_component(&mut rng);

        // initialize log probabilities
        let ln_weights: Vec<f64> =
            self.weights.iter().map(|&w| w.ln()).collect();
        let logps = Matrix::vtile(ln_weights, n_rows);

        self.accum_score_and_integrate_asgn(
            logps,
            n_cats + 1,
            RowAssignAlg::FiniteCpu,
            &mut rng,
        );
    }

    /// Use the improved slice algorithm to reassign the rows
    pub fn reassign_rows_slice(&mut self, mut rng: &mut impl Rng) {
        use crate::misc::sb_slice_extend;
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
        let n_cats = weights.len();

        for _ in 0..n_new_cats {
            self.append_empty_component(&mut rng);
        }

        // initialize truncated log probabilities
        let logps = {
            let mut values = Vec::with_capacity(weights.len() * self.n_rows());
            weights.iter().for_each(|w| {
                us.iter().for_each(|ui| {
                    let value = if w >= ui { 0.0 } else { NEG_INFINITY };
                    values.push(value);
                });
            });
            let matrix = Matrix::from_raw_parts(values, n_cats);
            debug_assert_eq!(matrix.n_cols(), us.len());
            debug_assert_eq!(matrix.n_rows(), weights.len());
            matrix
        };

        self.accum_score_and_integrate_asgn(
            logps,
            n_cats,
            RowAssignAlg::Slice,
            &mut rng,
        );
    }

    /// Resample the component weights
    ///
    /// # Note
    ///
    /// Used only for the FinteCpu and Slice algorithms
    #[inline]
    pub fn resample_weights(
        &mut self,
        add_empty_component: bool,
        mut rng: &mut impl Rng,
    ) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec).unwrap();
        self.weights = dir.draw(&mut rng)
    }

    /// Sequential adaptive merge-split (SAMS) row reassignment kernel
    pub fn reassign_rows_sams<R: Rng>(&mut self, rng: &mut R) {
        use rand::seq::IteratorRandom;

        let (i, j, zi, zj) = {
            let ixs = (0..self.n_rows()).choose_multiple(rng, 2);
            let i = ixs[0];
            let j = ixs[1];

            let zi = self.asgn.asgn[i];
            let zj = self.asgn.asgn[j];

            if zi < zj {
                (i, j, zi, zj)
            } else {
                (j, i, zj, zi)
            }
        };

        if zi == zj {
            self.sams_split(i, j, rng);
        } else {
            assert!(zi < zj);
            self.sams_merge(i, j, rng);
        }
        debug_assert!(self.asgn.validate().is_valid());
    }

    /// MCMC update on the CPR alpha parameter
    #[inline]
    pub fn update_alpha(&mut self, mut rng: &mut impl Rng) -> f64 {
        self.asgn
            .update_alpha(braid_consts::MH_PRIOR_ITERS, &mut rng)
    }

    /// Insert a new `Feature` into the `View`, but draw the feature
    /// components from the prior
    #[inline]
    pub fn init_feature(&mut self, mut ftr: ColModel, mut rng: &mut impl Rng) {
        let id = ftr.id();
        assert!(
            !self.ftrs.contains_key(&id),
            "Feature {} already in view",
            id
        );
        ftr.init_components(self.asgn.n_cats, &mut rng);
        ftr.reassign(&self.asgn, &mut rng);
        self.ftrs.insert(id, ftr);
    }

    /// Insert a new `Feature` into the `View`, but draw the feature components
    /// from the prior and redraw the data from those components.
    #[inline]
    pub(crate) fn geweke_init_feature(
        &mut self,
        mut ftr: ColModel,
        rng: &mut impl Rng,
    ) {
        let id = ftr.id();
        assert!(
            !self.ftrs.contains_key(&id),
            "Feature {} already in view",
            id
        );
        ftr.geweke_init(&self.asgn, rng);
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
        assert!(
            !self.ftrs.contains_key(&id),
            "Feature {} already in view",
            id
        );
        ftr.reassign(&self.asgn, &mut rng);

        self.ftrs.insert(id, ftr);
    }

    /// Remove and return the `Feature` with `id`. Returns `None` if the `id`
    /// is not found.
    #[inline]
    pub fn remove_feature(&mut self, id: usize) -> Option<ColModel> {
        self.ftrs.remove(&id)
    }

    // Delete the top/front n rows.
    pub fn del_rows_at<R: Rng>(&mut self, ix: usize, n: usize, rng: &mut R) {
        use crate::feature::FeatureHelper;

        assert!(ix + n <= self.n_rows());

        // Remove from suffstats, unassign, and drop components if singleton.
        // Get a list of the components that were removed so we can update the
        // assignment to preserve canonical order.
        (0..n).for_each(|_| {
            self.remove_row(ix);
            self.asgn.asgn.remove(ix);
        });

        // remove data from features
        for ftr in self.ftrs.values_mut() {
            (0..n).for_each(|_| {
                ftr.del_datum(ix);
            });
        }

        self.resample_weights(false, rng);
    }

    /// Remove all of the data from the features
    pub fn take_data(&mut self) -> BTreeMap<usize, FeatureData> {
        let mut data: BTreeMap<usize, FeatureData> = BTreeMap::new();
        self.ftrs.iter_mut().for_each(|(id, ftr)| {
            data.insert(*id, ftr.take_data());
        });
        data
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

// private view functions
impl View {
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
        });

        // The row might have been inserted into a new component, so we need to
        // re-sample the weights so the number of weights matches the number of
        // components
        self.resample_weights(false, &mut rng);
    }

    // Remove the row for the purposes of MCMC without deleting its data.
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

    /// Force component k to observe row_ix
    #[inline]
    fn force_observe_row(&mut self, row_ix: usize, k: usize) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.observe_datum(row_ix, k));
    }

    #[inline]
    fn reinsert_row(&mut self, row_ix: usize, mut rng: &mut impl Rng) {
        let k_new = if self.asgn.n_cats == 0 {
            // If empty, assign to category zero
            debug_assert!(self.ftrs.values().all(|f| f.k() == 0));
            self.append_empty_component(&mut rng);
            0
        } else {
            // If not empty, do a Gibbs step
            let mut logps: Vec<f64> = Vec::with_capacity(self.asgn.n_cats + 1);
            self.asgn.counts.iter().enumerate().for_each(|(k, &ct)| {
                logps.push(
                    (ct as f64).ln() + self.predictive_score_at(row_ix, k),
                );
            });

            logps.push(self.asgn.alpha.ln() + self.singleton_score(row_ix));

            let k_new = ln_pflip(&logps, 1, false, &mut rng)[0];

            if k_new == self.asgn.n_cats {
                self.append_empty_component(&mut rng);
            }

            k_new
        };

        self.observe_row(row_ix, k_new);
        self.asgn.reassign(row_ix, k_new);
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
        n_cats: usize,
        mut rng: &mut impl Rng,
    ) {
        // Returns the unused category indices in descending order so that
        // removing the unused components and reindexing requires less
        // bookkeeping
        let unused_cats = unused_components(n_cats, &new_asgn_vec);

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

    fn set_asgn<R: Rng>(&mut self, asgn: Assignment, rng: &mut R) {
        self.asgn = asgn;
        self.resample_weights(false, rng);
        for ftr in self.ftrs.values_mut() {
            ftr.reassign(&self.asgn, rng)
        }
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

    #[inline]
    fn get_sams_indices<R: Rng>(
        &self,
        zi: usize,
        zj: usize,
        calc_reverse: bool,
        rng: &mut R,
    ) -> Vec<usize> {
        if calc_reverse {
            // Get the indices of the columns assigned to the clusters that
            // were split
            self.asgn
                .asgn
                .iter()
                .enumerate()
                .filter_map(
                    |(ix, &z)| {
                        if z == zi || z == zj {
                            Some(ix)
                        } else {
                            None
                        }
                    },
                )
                .collect()
        } else {
            // Get the indices of the columns assigned to the cluster to split
            let mut row_ixs: Vec<usize> = self
                .asgn
                .asgn
                .iter()
                .enumerate()
                .filter_map(|(ix, &z)| if z == zi { Some(ix) } else { None })
                .collect();

            row_ixs.shuffle(rng);
            row_ixs
        }
    }

    fn sams_merge<R: Rng>(&mut self, i: usize, j: usize, rng: &mut R) {
        use crate::assignment::lcrp;
        use std::cmp::Ordering;

        let zi = self.asgn.asgn[i];
        let zj = self.asgn.asgn[j];

        let (logp_spt, logq_spt, ..) = self.propose_split(i, j, true, rng);

        let asgn = {
            let zs = self
                .asgn
                .asgn
                .iter()
                .map(|&z| match z.cmp(&zj) {
                    Ordering::Equal => zi,
                    Ordering::Greater => z - 1,
                    Ordering::Less => z,
                })
                .collect();

            AssignmentBuilder::from_vec(zs)
                .with_prior(self.asgn.prior.clone())
                .with_alpha(self.asgn.alpha)
                .seed_from_rng(rng)
                .build()
                .unwrap()
        };

        self.append_empty_component(rng);
        asgn.asgn.iter().enumerate().for_each(|(ix, &z)| {
            if z == zi {
                self.force_observe_row(ix, self.n_cats());
            }
        });

        let logp_mrg = self.logm(self.n_cats())
            + lcrp(asgn.len(), &asgn.counts, asgn.alpha);

        self.drop_component(self.n_cats());

        if rng.gen::<f64>().ln() < logp_mrg - logp_spt + logq_spt {
            self.set_asgn(asgn, rng)
        }
    }

    fn sams_split<R: Rng>(&mut self, i: usize, j: usize, rng: &mut R) {
        use crate::assignment::lcrp;

        let zi = self.asgn.asgn[i];

        let logp_mrg = self.logm(zi)
            + lcrp(self.asgn.len(), &self.asgn.counts, self.asgn.alpha);
        let (logp_spt, logq_spt, asgn_opt) =
            self.propose_split(i, j, false, rng);

        let asgn = asgn_opt.unwrap();

        if rng.gen::<f64>().ln() < logp_spt - logp_mrg - logq_spt {
            self.set_asgn(asgn, rng)
        }
    }

    // TODO: this is a long-ass bitch
    fn propose_split<R: Rng>(
        &mut self,
        i: usize,
        j: usize,
        calc_reverse: bool,
        rng: &mut R,
    ) -> (f64, f64, Option<Assignment>) {
        use crate::assignment::lcrp;

        let zi = self.asgn.asgn[i];
        let zj = self.asgn.asgn[j];

        self.append_empty_component(rng);
        self.append_empty_component(rng);

        let zi_tmp = self.asgn.n_cats;
        let zj_tmp = zi_tmp + 1;

        self.force_observe_row(i, zi_tmp);
        self.force_observe_row(j, zj_tmp);

        let mut tmp_z: Vec<usize> = {
            // mark everything assigned to the split cluster as unassigned (-1)
            let mut zs: Vec<usize> = self
                .asgn
                .iter()
                .map(|&z| if z == zi { std::usize::MAX } else { z })
                .collect();
            zs[i] = zi_tmp;
            zs[j] = zj_tmp;
            zs
        };

        let row_ixs = self.get_sams_indices(zi, zj, calc_reverse, rng);

        let mut logq: f64 = 0.0;
        let mut nk_i: f64 = 1.0;
        let mut nk_j: f64 = 1.0;

        row_ixs
            .iter()
            .filter(|&&ix| !(ix == i || ix == j))
            .for_each(|&ix| {
                let logp_zi = nk_i.ln() + self.predictive_score_at(ix, zi_tmp);
                let logp_zj = nk_j.ln() + self.predictive_score_at(ix, zj_tmp);
                let lognorm = logaddexp(logp_zi, logp_zj);

                let assign_to_zi = if calc_reverse {
                    self.asgn.asgn[ix] == zi
                } else {
                    rng.gen::<f64>().ln() < logp_zi - lognorm
                };

                if assign_to_zi {
                    logq += logp_zi - lognorm;
                    self.force_observe_row(ix, zi_tmp);
                    nk_i += 1.0;
                    tmp_z[ix] = zi_tmp;
                } else {
                    logq += logp_zj - lognorm;
                    self.force_observe_row(ix, zj_tmp);
                    nk_j += 1.0;
                    tmp_z[ix] = zj_tmp;
                }
            });

        let mut logp = self.logm(zi_tmp) + self.logm(zj_tmp);

        let asgn = if calc_reverse {
            logp += lcrp(self.asgn.len(), &self.asgn.counts, self.asgn.alpha);
            None
        } else {
            tmp_z.iter_mut().for_each(|z| {
                if *z == zi_tmp {
                    *z = zi;
                } else if *z == zj_tmp {
                    *z = self.n_cats();
                }
            });

            let asgn = AssignmentBuilder::from_vec(tmp_z)
                .with_prior(self.asgn.prior.clone())
                .with_alpha(self.asgn.alpha)
                .seed_from_rng(rng)
                .build()
                .unwrap();

            logp += lcrp(asgn.len(), &asgn.counts, asgn.alpha);
            Some(asgn)
        };

        // delete the last component twice since we appended two components
        self.drop_component(self.n_cats());
        self.drop_component(self.n_cats());

        (logp, logq, asgn)
    }

    fn accum_score_and_integrate_asgn(
        &mut self,
        mut logps: Matrix<f64>,
        n_cats: usize,
        row_alg: RowAssignAlg,
        rng: &mut impl Rng,
    ) {
        // TODO: parallelize over rows_mut somehow?
        logps.rows_mut().enumerate().for_each(|(k, logp)| {
            self.ftrs.values().for_each(|ftr| {
                ftr.accum_score(logp, k);
            })
        });

        // Implicit transpose does not change the memory layout, just the
        // indexing.
        let logps = logps.implicit_transpose();
        debug_assert_eq!(logps.n_rows(), self.n_rows());

        let new_asgn_vec = match row_alg {
            RowAssignAlg::Slice => massflip_slice_mat_par(&logps, rng),
            _ => massflip(&logps, rng),
        };

        self.integrate_finite_asgn(new_asgn_vec, n_cats, rng);
    }
}

// Geweke
// ======
/// Configuration of the Geweke test on Views
pub struct ViewGewekeSettings {
    /// The number of columns/features in the view
    pub n_cols: usize,
    /// The number of rows in the view
    pub n_rows: usize,
    /// Column model types
    pub cm_types: Vec<FType>,
    /// Which transitions to run
    pub transitions: Vec<ViewTransition>,
}

impl ViewGewekeSettings {
    pub fn new(n_rows: usize, cm_types: Vec<FType>) -> Self {
        ViewGewekeSettings {
            n_rows,
            n_cols: cm_types.len(),
            cm_types,
            // XXX: You HAVE to run component params update explicitly for gibbs
            // and SAMS reassignment kernels because these algorithms do not do
            // parameter updates explicitly (they marginalize over the component
            // parameters) and the data resample relies on the component
            // parameters.
            transitions: vec![
                ViewTransition::RowAssignment(RowAssignAlg::Slice),
                ViewTransition::FeaturePriors,
                ViewTransition::ComponentParams,
                ViewTransition::Alpha,
            ],
        }
    }

    pub fn do_row_asgn_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, ViewTransition::RowAssignment(_)))
    }

    pub fn do_alpha_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, ViewTransition::Alpha))
    }
}

fn view_geweke_asgn(
    n_rows: usize,
    do_alpha_transition: bool,
    do_row_asgn_transition: bool,
) -> AssignmentBuilder {
    let mut bldr = AssignmentBuilder::new(n_rows).with_geweke_prior();

    if !do_row_asgn_transition {
        bldr = bldr.flat();
    }

    if !do_alpha_transition {
        bldr = bldr.with_alpha(1.0);
    }

    bldr
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

        let asgn = view_geweke_asgn(
            settings.n_rows,
            settings.do_alpha_transition(),
            settings.do_row_asgn_transition(),
        )
        .seed_from_rng(&mut rng)
        .build()
        .unwrap();

        // this function sets up dummy features that we can properly populate with
        // Feature.geweke_init in the next loop
        let mut ftrs = gen_geweke_col_models(
            &settings.cm_types,
            settings.n_rows,
            do_ftr_prior_transition,
            &mut rng,
        );

        let ftrs: BTreeMap<_, _> = ftrs
            .drain(..)
            .enumerate()
            .map(|(id, mut ftr)| {
                ftr.geweke_init(&asgn, &mut rng);
                (id, ftr)
            })
            .collect();

        View {
            ftrs,
            weights: asgn.weights(),
            asgn,
        }
    }

    fn geweke_step(
        &mut self,
        settings: &ViewGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        self.step(&settings.transitions, &mut rng);
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
    pub n_cats: Option<usize>,
    /// The CRP alpha
    pub alpha: Option<f64>,
    /// The summary for each column/feature.
    pub cols: Vec<(usize, GewekeColumnSummary)>,
}

impl From<&GewekeViewSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeViewSummary) -> BTreeMap<String, f64> {
        let mut map: BTreeMap<String, f64> = BTreeMap::new();
        if let Some(n_cats) = value.n_cats {
            map.insert("n_cats".into(), n_cats as f64);
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
        let col_settings = ColumnGewekeSettings::new(
            self.asgn.clone(),
            settings.transitions.clone(),
        );

        GewekeViewSummary {
            n_cats: if settings.do_row_asgn_transition() {
                Some(self.n_cats())
            } else {
                None
            },
            alpha: if settings.do_alpha_transition() {
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

    use crate::component::ConjugateComponent;
    use crate::feature::Column;
    use braid_data::SparseContainer;
    use braid_stats::prior::nix::NixHyper;
    use braid_stats::rv::dist::{Gaussian, NormalInvChiSquared};

    fn gen_col<R: Rng>(id: usize, n: usize, mut rng: &mut R) -> ColModel {
        let gauss = Gaussian::new(0.0, 1.0).unwrap();
        let data_vec: Vec<f64> = (0..n).map(|_| gauss.draw(&mut rng)).collect();
        let data = SparseContainer::from(data_vec);
        let hyper = NixHyper::default();
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);

        let ftr = Column::new(id, data, prior, hyper);
        ColModel::Continuous(ftr)
    }

    fn gen_gauss_view<R: Rng>(n: usize, mut rng: &mut R) -> View {
        let mut ftrs: Vec<ColModel> = vec![];
        ftrs.push(gen_col(0, n, &mut rng));
        ftrs.push(gen_col(1, n, &mut rng));
        ftrs.push(gen_col(2, n, &mut rng));
        ftrs.push(gen_col(3, n, &mut rng));

        Builder::new(n)
            .features(ftrs)
            .seed_from_rng(&mut rng)
            .build()
    }

    fn extract_components(
        view: &View,
    ) -> Vec<Vec<ConjugateComponent<f64, Gaussian, NormalInvChiSquared>>> {
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

    macro_rules! test_singleton_reassign {
        ($alg:expr, $fn:ident) => {
            #[test]
            fn $fn() {
                let mut rng = rand::thread_rng();
                let mut view = gen_gauss_view(1, &mut rng);
                view.reassign($alg, &mut rng);
            }
        };
    }

    test_singleton_reassign!(
        RowAssignAlg::FiniteCpu,
        singleton_reassign_smoke_finite_cpu
    );

    test_singleton_reassign!(RowAssignAlg::Sams, singleton_reassign_smoke_sams);

    test_singleton_reassign!(
        RowAssignAlg::Slice,
        singleton_reassign_smoke_slice
    );

    test_singleton_reassign!(
        RowAssignAlg::Gibbs,
        singleton_reassign_smoke_gibbs
    );

    #[test]
    fn seeding_view_works() {
        let view_1 = {
            let mut rng = Xoshiro256Plus::seed_from_u64(1338);
            gen_gauss_view(1000, &mut rng)
        };

        let view_2 = {
            let mut rng = Xoshiro256Plus::seed_from_u64(1338);
            gen_gauss_view(1000, &mut rng)
        };

        assert_eq!(view_1.asgn.asgn, view_2.asgn.asgn);
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

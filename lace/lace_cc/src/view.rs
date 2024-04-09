use std::collections::BTreeMap;

use lace_data::{Datum, FeatureData};
use lace_stats::assignment::Assignment;
use lace_stats::prior_process::PriorProcess;
use lace_stats::rv::dist::Dirichlet;
use lace_stats::rv::traits::Rv;
use lace_utils::{unused_components, Matrix};
use rand::Rng;
use serde::{Deserialize, Serialize};

// use crate::cc::feature::geweke::{gen_geweke_col_models, ColumnGewekeSettings};
use crate::alg::RowAssignAlg;
use crate::constrain::RowGibbsInfo;
use crate::feature::{ColModel, Feature};
use crate::transition::ViewTransition;

mod builder;
pub mod geweke;
mod gibbs;
mod sams;
mod slice;

pub use builder::Builder;

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
    pub prior_process: PriorProcess,
    /// The weights of each category
    pub weights: Vec<f64>,
}

impl View {
    pub fn asgn(&self) -> &Assignment {
        &self.prior_process.asgn
    }

    pub fn asgn_mut(&mut self) -> &mut Assignment {
        &mut self.prior_process.asgn
    }

    /// The number of rows in the `View`
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.asgn().len()
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
        self.asgn().n_cats
    }

    // Extend the columns by a number of cells, increasing the total number of
    // rows. The added entries will be empty.
    pub fn extend_cols(&mut self, n_rows: usize) {
        (0..n_rows).for_each(|_| self.asgn_mut().push_unassigned());
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
        let k = self.asgn().asgn[row_ix];
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

        let k = self.asgn().asgn[row_ix];
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
                ViewTransition::PriorProcessParams => {
                    self.update_prior_process_params(&mut rng);
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
            ViewTransition::PriorProcessParams,
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
            RowAssignAlg::Slice => self.reassign_rows_slice(&(), &mut rng),
            RowAssignAlg::Gibbs => self.reassign_rows_gibbs(&mut rng),
            RowAssignAlg::Sams => self.reassign_rows_sams(&(), &mut rng),
        }
    }

    /// Use the finite approximation (on the CPU) to reassign the rows
    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut impl Rng) {
        let n_cats = self.n_cats();
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
            &(),
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
        let dirvec =
            self.prior_process.weight_vec_unnormed(add_empty_component);

        if dirvec.iter().any(|&p| p < 0.0) {
            eprintln!("{:?}", dirvec);
            eprintln!("{:?}\n", self.prior_process.process);
        }

        let dir = Dirichlet::new(dirvec).unwrap();
        self.weights = dir.draw(&mut rng)
    }

    /// MCMC update on the CPR alpha parameter
    #[inline]
    pub fn update_prior_process_params(&mut self, rng: &mut impl Rng) -> f64 {
        self.prior_process.update_params(rng);
        // FIXME: should be the new likelihood
        0.0
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
        ftr.init_components(self.asgn().n_cats, &mut rng);
        ftr.reassign(self.asgn(), &mut rng);
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
        ftr.geweke_init(self.asgn(), rng);
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
        ftr.reassign(self.asgn(), &mut rng);

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
            self.asgn_mut().asgn.remove(ix);
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
            ftr.reassign(&self.prior_process.asgn, &mut rng);
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
            .asgn()
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
            self.reinsert_row(row_ix, RowGibbsInfo::default(), &(), &mut rng);
        });

        // The row might have been inserted into a new component, so we need to
        // re-sample the weights so the number of weights matches the number of
        // components
        self.resample_weights(false, &mut rng);
    }

    /// Force component k to observe row_ix
    fn force_observe_row(&mut self, row_ix: usize, k: usize) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.observe_datum(row_ix, k));
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

        self.asgn_mut()
            .set_asgn(new_asgn_vec)
            .expect("new asgn is invalid");
        self.resample_weights(false, &mut rng);
        for ftr in self.ftrs.values_mut() {
            ftr.reassign(&self.prior_process.asgn, &mut rng)
        }
    }

    fn set_asgn<R: Rng>(&mut self, asgn: Assignment, rng: &mut R) {
        self.prior_process.asgn = asgn;
        self.resample_weights(false, rng);
        for ftr in self.ftrs.values_mut() {
            ftr.reassign(&self.prior_process.asgn, rng)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::component::ConjugateComponent;
    use crate::feature::Column;

    use lace_data::SparseContainer;
    use lace_stats::prior::nix::NixHyper;
    use lace_stats::rv::dist::{Gaussian, NormalInvChiSquared};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

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
        let features: Vec<ColModel> = vec![
            gen_col(0, n, &mut rng),
            gen_col(1, n, &mut rng),
            gen_col(2, n, &mut rng),
            gen_col(3, n, &mut rng),
        ];

        Builder::new(n)
            .features(features)
            .seed_from_rng(&mut rng)
            .build()
    }

    fn extract_components(
        view: &View,
    ) -> Vec<Vec<ConjugateComponent<f64, Gaussian, NormalInvChiSquared>>> {
        view.ftrs
            .values()
            .map(|ftr| {
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

        assert_eq!(view_1.asgn().asgn, view_2.asgn().asgn);
    }

    #[test]
    fn extend_cols_adds_empty_unassigned_rows() {
        let mut rng = rand::thread_rng();
        let mut view = gen_gauss_view(10, &mut rng);

        let components_start = extract_components(&view);

        view.extend_cols(2);

        assert_eq!(view.asgn().asgn.len(), 12);
        assert_eq!(view.asgn().asgn[10], usize::max_value());
        assert_eq!(view.asgn().asgn[11], usize::max_value());

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

        let view_ix_start = view.asgn().asgn[2];
        let component_start = components_start[3][view_ix_start].clone();

        view.insert_datum(2, 3, Datum::Continuous(20.22));

        let components_end = extract_components(&view);
        let view_ix_end = view.asgn().asgn[2];
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

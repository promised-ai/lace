extern crate rand;
extern crate serde;

use std::collections::BTreeMap;

use self::rand::Rng;
use cc::column_model::gen_geweke_col_models;
use cc::container::FeatureData;
use cc::{Assignment, ColModel, DType, FType, Feature};
use dist::traits::RandomVariate;
use dist::Dirichlet;
use geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use misc::{massflip, transpose, unused_components};

// number of interations used by the MH sampler when updating paramters
const N_MH_ITERS: usize = 50;

/// View is a multivariate generalization of the standard Diriclet-process
/// mixture model (DPGMM). `View` captures a joint distibution over its
/// columns by assuming the columns are dependent.
#[derive(Serialize, Deserialize, Clone)]
pub struct View {
    pub ftrs: BTreeMap<usize, ColModel>,
    pub asgn: Assignment,
    pub weights: Vec<f64>,
    pub alpha: f64,
}

unsafe impl Send for View {}
unsafe impl Sync for View {}

/// The MCMC algorithm to use for row reassignment
#[derive(Clone, Copy)]
pub enum RowAssignAlg {
    /// CPU-parallelized finite Dirichlet appproximation
    FiniteCpu,
    /// OpenCL GPU-parallelized finite Dirichlet appproximation
    FiniteGpu,
    /// Sequential importance samplint split-merge
    SplitMerge,
}

impl View {
    /// Construct a View from a vector of `Box`ed `Feature`s
    pub fn new(ftrs: Vec<ColModel>, _alpha: f64, mut rng: &mut Rng) -> View {
        let nrows = ftrs[0].len();
        let asgn = Assignment::from_prior(nrows, &mut rng);
        View::with_assignment(ftrs, asgn, &mut rng)
    }

    pub fn with_assignment(
        mut ftrs: Vec<ColModel>,
        asgn: Assignment,
        mut rng: &mut Rng,
    ) -> Self {
        let alpha = asgn.alpha;
        let weights = asgn.weights();
        let k = asgn.ncats;
        for ftr in ftrs.iter_mut() {
            ftr.init_components(k, &mut rng);
        }

        let mut ftrs_tree = BTreeMap::new();
        for ftr in ftrs.drain(0..) {
            ftrs_tree.insert(ftr.id(), ftr);
        }

        View {
            ftrs: ftrs_tree,
            asgn: asgn,
            alpha: alpha,
            weights: weights,
        }
    }

    // No views
    pub fn flat(n: usize) -> View {
        let alpha = 1.0;
        let asgn = Assignment::flat(n, alpha);
        let ftrs: BTreeMap<usize, ColModel> = BTreeMap::new();
        View {
            ftrs: ftrs,
            asgn: asgn,
            alpha: alpha,
            weights: vec![1.0],
        }
    }

    // No views
    pub fn empty(n: usize, alpha: f64, mut rng: &mut Rng) -> View {
        let asgn = Assignment::draw(n, alpha, &mut rng);
        let ftrs: BTreeMap<usize, ColModel> = BTreeMap::new();
        let weights = asgn.weights();
        View {
            ftrs: ftrs,
            asgn: asgn,
            alpha: alpha,
            weights: weights,
        }
    }

    /// Returns the number of rows in the `View`
    pub fn nrows(&self) -> usize {
        self.asgn.asgn.len()
    }

    /// Returns the number of columns in the `View`
    pub fn ncols(&self) -> usize {
        self.ftrs.len()
    }

    pub fn ncats(&self) -> usize {
        self.asgn.ncats
    }

    pub fn logp_at(&self, row_ix: usize, col_ix: usize) -> Option<f64> {
        let k = self.asgn.asgn[row_ix];
        self.ftrs[&col_ix].logp_at(row_ix, k)
    }

    pub fn get_datum(&self, row_ix: usize, col_ix: usize) -> Option<DType> {
        if self.ftrs.contains_key(&col_ix) {
            Some(self.ftrs[&col_ix].get_datum(row_ix))
        } else {
            None
        }
    }

    /// Update the state of the `View` by running the `View` MCMC transitions
    /// `n_iter` times.
    pub fn update(
        &mut self,
        n_iter: usize,
        alg: RowAssignAlg,
        mut rng: &mut Rng,
    ) {
        for _ in 0..n_iter {
            self.reassign(alg.clone(), &mut rng);
            self.update_alpha(&mut rng);
            self.update_prior_params(&mut rng);
        }
    }

    pub fn update_prior_params(&mut self, mut rng: &mut Rng) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.update_prior_params(&mut rng));
    }

    pub fn update_component_params(&mut self, mut rng: &mut Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.update_components(&self.asgn, &mut rng);
        }
    }

    /// Reassign the rows to categories
    pub fn reassign(&mut self, alg: RowAssignAlg, mut rng: &mut Rng) {
        match alg {
            RowAssignAlg::FiniteGpu => self.reassign_rows_finite_gpu(&mut rng),
            RowAssignAlg::FiniteCpu => self.reassign_rows_finite_cpu(&mut rng),
            RowAssignAlg::SplitMerge => {
                self.reassign_rows_split_merge(&mut rng)
            }
        }
    }

    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut Rng) {
        let ncats = self.asgn.ncats;
        let nrows = self.nrows();

        self.resample_weights(true, &mut rng);
        self.append_empty_component(&mut rng);

        // initialize log probabilities
        // TODO: This way of initialization may be slow
        let mut logps: Vec<Vec<f64>> = vec![vec![0.0; nrows]; ncats + 1];
        for (k, w) in self.weights.iter().enumerate() {
            logps[k] = vec![w.ln(); nrows];
        }

        for k in 0..(ncats + 1) {
            for (_, ftr) in &self.ftrs {
                ftr.accum_score(&mut logps[k], k);
            }
        }

        let logps_t = transpose(&logps);
        let new_asgn_vec = massflip(logps_t, &mut rng);

        self.integrate_finite_asgn(new_asgn_vec);

        // We resample the weights w/o the CRP alpha appended so that the
        // number of weights matches the number of components
        self.resample_weights(false, &mut rng);
        // XXX: if update_component_params is not called the components in the
        // features will not reflect the assignment. Reassign does not modify
        // features, it only modifies the assignment.
        self.update_component_params(&mut rng);
    }

    pub fn resample_weights(
        &mut self,
        add_empty_component: bool,
        mut rng: &mut Rng,
    ) {
        let dirvec = self.asgn.dirvec(add_empty_component);
        let dir = Dirichlet::new(dirvec);
        self.weights = dir.draw(&mut rng)
    }

    pub fn reassign_rows_finite_gpu(&mut self, _rng: &mut Rng) {
        unimplemented!();
    }

    pub fn reassign_rows_split_merge(&mut self, _rng: &mut Rng) {
        // Naive, SIS split-merge
        // ======================
        //
        // 1. choose two columns, i and j
        // 2. If i == j, split(i, j) else merge(i, j)
        //
        // Split
        // -----
        // Def. k := the component to which i and j are currently assigned
        // Def. x_k := all the data assigned to component k
        // 1. Create a component with x_k
        // 2. Create two components: one with the datum at i and one with the
        //    datum at j.
        // 3. Assign the remaning data to components i or j via SIS
        // 4. Do Proposal
        //
        // Merge
        // ----
        // 1. Create a component with x_i and x_j combined
        // 2. Create two components: one with with datum i and one with datum j
        // 3. Compute the reverse probability of the given assignment of a
        //    split
        // 4. Compute the MH acceptance
        unimplemented!();
    }

    pub fn update_alpha(&mut self, mut rng: &mut Rng) {
        self.asgn.update_alpha(N_MH_ITERS, &mut rng);
    }

    fn append_empty_component(&mut self, mut rng: &mut Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.append_empty_component(&mut rng);
        }
    }

    fn drop_component(&mut self, k: usize) {
        for ftr in self.ftrs.values_mut() {
            ftr.drop_component(k);
        }
    }

    // Cleanup functions
    fn integrate_finite_asgn(&mut self, mut new_asgn_vec: Vec<usize>) {
        // 1. Find unused components
        // let ncats = self.asgn.ncats;
        // let all_cats: HashSet<_> = HashSet::from_iter(0..ncats);
        // let used_cats = HashSet::from_iter(new_asgn_vec.iter().cloned());
        // let mut unused_cats: Vec<&usize> = all_cats.difference(&used_cats)
        //                                            .collect();
        // unused_cats.sort();
        // // needs to be in reverse order, because we want to remove the
        // // higher-indexed components first to minimize bookkeeping.
        // unused_cats.reverse();
        let unused_cats = unused_components(self.asgn.ncats, &new_asgn_vec);

        for k in unused_cats {
            self.drop_component(k);
            for z in new_asgn_vec.iter_mut() {
                if *z > k {
                    *z -= 1
                };
            }
        }
        self.asgn = Assignment::from_vec(new_asgn_vec, self.alpha);
        assert!(self.asgn.validate().is_valid());
    }

    /// Insert a new `Feature` into the `View`, but draw the feature
    /// components from the prior
    pub fn init_feature(&mut self, mut ftr: ColModel, mut rng: &mut Rng) {
        let id = ftr.id();
        if self.ftrs.contains_key(&id) {
            panic!("Feature {} already in view", id);
        }
        ftr.init_components(self.asgn.ncats, &mut rng);
        self.ftrs.insert(id, ftr);
    }

    /// Insert a new `Feature` into the `View`
    pub fn insert_feature(&mut self, mut ftr: ColModel, mut rng: &mut Rng) {
        let id = ftr.id();
        if self.ftrs.contains_key(&id) {
            panic!("Feature {} already in view", id);
        }
        ftr.reassign(&self.asgn, &mut rng);
        self.ftrs.insert(id, ftr);
    }

    /// Remove and return the `Feature` with `id`. Returns `None` if the `id`
    /// is not found.
    pub fn remove_feature(&mut self, id: usize) -> Option<ColModel> {
        self.ftrs.remove(&id)
    }

    pub fn take_data(&mut self) -> BTreeMap<usize, FeatureData> {
        let mut data: BTreeMap<usize, FeatureData> = BTreeMap::new();
        self.ftrs.iter_mut().for_each(|(id, ftr)| {
            data.insert(*id, ftr.take_data());
        });
        data
    }
}

// Geweke
// ======
pub struct ViewGewekeSettings {
    /// The number of columns/features in the view
    pub ncols: usize,
    /// The number of rows in the view
    pub nrows: usize,
    /// The row reassignment algorithm
    pub row_alg: RowAssignAlg,
    /// Column model types
    pub cm_types: Vec<FType>,
}

impl ViewGewekeSettings {
    pub fn new(nrows: usize, cm_types: Vec<FType>) -> Self {
        ViewGewekeSettings {
            nrows: nrows,
            ncols: cm_types.len(),
            row_alg: RowAssignAlg::FiniteCpu,
            cm_types: cm_types,
        }
    }
}

impl GewekeModel for View {
    // FIXME: need nrows, ncols, and algorithm specification
    fn geweke_from_prior(
        settings: &ViewGewekeSettings,
        mut rng: &mut Rng,
    ) -> View {
        let ftrs =
            gen_geweke_col_models(&settings.cm_types, settings.nrows, &mut rng);
        View::new(ftrs, 1.0, &mut rng)
    }

    fn geweke_step(
        &mut self,
        settings: &ViewGewekeSettings,
        mut rng: &mut Rng,
    ) {
        self.update(1, settings.row_alg.clone(), &mut rng);
    }
}

impl GewekeResampleData for View {
    type Settings = ViewGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        _s: Option<&ViewGewekeSettings>,
        rng: &mut Rng,
    ) {
        for ftr in self.ftrs.values_mut() {
            ftr.geweke_resample_data(Some(&self.asgn), rng);
        }
    }
}

impl GewekeSummarize for View {
    fn geweke_summarize(&self) -> BTreeMap<String, f64> {
        let mut summary: BTreeMap<String, f64> = BTreeMap::new();

        summary.insert(String::from("ncats"), self.ncats() as f64);

        for (_, ftr) in &self.ftrs {
            // TODO: add column id to map key
            let mut ftr_summary = {
                let id: usize = ftr.id();
                let summary = ftr.geweke_summarize();
                let label: String = format!("Feature {} ", id);
                let mut relabled_summary = BTreeMap::new();
                for (key, value) in summary {
                    relabled_summary.insert(label.clone() + &key, value);
                }
                relabled_summary
            };
            summary.append(&mut ftr_summary);
        }
        summary
    }
}

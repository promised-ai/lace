use std::collections::BTreeMap;
use std::f64::NEG_INFINITY;

use braid_flippers::massflip_slice;
use braid_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use braid_utils::misc::{transpose, unused_components};
use rand::{seq::SliceRandom as _, FromEntropy, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::{Dirichlet, Gamma};
use rv::misc::ln_pflip;
use rv::traits::Rv;
use serde::{Deserialize, Serialize};

use crate::cc::feature::geweke::{gen_geweke_col_models, ColumnGewekeSettings};
use crate::cc::feature::FeatureData;
use crate::cc::transition::ViewTransition;
use crate::cc::{
    AppendRowsData, Assignment, AssignmentBuilder, ColModel, Datum, FType,
    Feature, RowAssignAlg,
};
use crate::misc::massflip;
use crate::result;

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
    alpha_prior: Option<Gamma>,
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
    pub fn with_alpha_prior(
        mut self,
        alpha_prior: Gamma,
    ) -> result::Result<Self> {
        if self.asgn.is_some() {
            let err = result::Error::new(
                result::ErrorKind::AlreadyExistsError,
                String::from("Cannot add alpha_prior once Assignment added"),
            );
            Err(err)
        } else {
            self.alpha_prior = Some(alpha_prior);
            Ok(self)
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
    pub fn from_rng<R: Rng>(mut self, rng: &mut R) -> Self {
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
                        .from_rng(&mut rng)
                        .build()
                        .unwrap()
                } else {
                    AssignmentBuilder::new(self.nrows)
                        .with_prior(self.alpha_prior.unwrap())
                        .from_rng(&mut rng)
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
    pub fn nrows(&self) -> usize {
        self.asgn.asgn.len()
    }

    /// The number of columns in the `View`
    pub fn ncols(&self) -> usize {
        self.ftrs.len()
    }

    /// The number of columns/features
    pub fn len(&self) -> usize {
        self.ncols()
    }

    /// returns true if there are no features
    pub fn is_empty(&self) -> bool {
        self.ncols() == 0
    }

    /// The number of categories
    pub fn ncats(&self) -> usize {
        self.asgn.ncats
    }

    /// The current value of the CPR alpha parameter
    pub fn alpha(&self) -> f64 {
        self.asgn.alpha
    }

    pub fn append_rows(
        &mut self,
        new_rows: Vec<&AppendRowsData>,
        mut rng: &mut impl Rng,
    ) {
        assert_eq!(self.ncols(), new_rows.len());

        let nrows = self.nrows();
        let n_new_rows = new_rows[0].len();
        for row_ix in 0..n_new_rows {
            self.asgn.push_unassigned();
            for ftr_rows in new_rows.iter() {
                self.ftrs
                    .get_mut(&ftr_rows.col_ix)
                    .unwrap()
                    .append_datum(ftr_rows.data[row_ix].clone())
            }
            // Insert row by Gibbs
            self.reinsert_row(nrows + row_ix, &mut rng)
        }
    }

    /// Get the log PDF/PMF of the datum at `row_ix` in feature `col_ix`
    pub fn logp_at(&self, row_ix: usize, col_ix: usize) -> Option<f64> {
        let k = self.asgn.asgn[row_ix];
        self.ftrs[&col_ix].logp_at(row_ix, k)
    }

    /// The probability of the row at `row_ix` belonging to cluster `k` given
    /// the data already assigned to category `k` with all component parameters
    /// marginalized away
    pub fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        self.ftrs
            .values()
            .fold(0.0, |acc, ftr| acc + ftr.predictive_score_at(row_ix, k))
    }

    /// The marginal likelihood of `row_ix`
    pub fn singleton_score(&self, row_ix: usize) -> f64 {
        self.ftrs
            .values()
            .fold(0.0, |acc, ftr| acc + ftr.singleton_score(row_ix))
    }

    /// get the datum at `row_ix` under the feature with id `col_ix`
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
        transitions: &Vec<ViewTransition>,
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
    pub fn update(
        &mut self,
        n_iters: usize,
        alg: RowAssignAlg,
        transitions: &Vec<ViewTransition>,
        mut rng: &mut impl Rng,
    ) {
        (0..n_iters).for_each(|_| self.step(alg, &transitions, &mut rng))
    }

    /// Update the prior parameters on each feature
    pub fn update_prior_params(&mut self, mut rng: &mut impl Rng) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.update_prior_params(&mut rng));
    }

    /// Update the component parameters in each feature
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

    fn remove_row(&mut self, row_ix: usize) {
        let k = self.asgn.asgn[row_ix];
        let is_singleton = self.asgn.counts[k] == 1;
        self.forget_row(row_ix, k);
        self.asgn.unassign(row_ix);

        if is_singleton {
            self.drop_component(k);
        }
    }

    fn reinsert_row(&mut self, row_ix: usize, mut rng: &mut impl Rng) {
        let mut logps = self.asgn.log_dirvec(true);
        (0..self.asgn.ncats).for_each(|k| {
            logps[k] += self.predictive_score_at(row_ix, k);
        });
        logps[self.asgn.ncats] += self.singleton_score(row_ix);

        let k_new = ln_pflip(&logps, 1, false, &mut rng)[0];
        if k_new == self.asgn.ncats {
            self.append_empty_component(&mut rng);
        }

        self.observe_row(row_ix, k_new);
        self.asgn
            .reassign(row_ix, k_new)
            .expect("Failed to reassign");
    }

    /// Use the standard Gibbs kernel to reassign the rows
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
    }

    /// Use the finite approximation (on the CPU) to reassign the rows
    pub fn reassign_rows_finite_cpu(&mut self, mut rng: &mut impl Rng) {
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

        let udist = rand::distributions::Open01;

        let weights: Vec<f64> = {
            let dirvec = self.asgn.dirvec(true);
            let dir = Dirichlet::new(dirvec.clone()).unwrap();
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

        let weights =
            sb_slice_extend(weights.clone(), self.asgn.alpha, u_star, &mut rng)
                .expect("Failed to break sticks");

        let n_new_cats = weights.len() - self.weights.len();
        let ncats = weights.len();

        for _ in 0..n_new_cats {
            self.append_empty_component(&mut rng);
        }

        // initialize truncated log probabilities
        let logps: Vec<Vec<f64>> = weights
            .iter()
            .map(|w| {
                let lpk: Vec<f64> = us
                    .iter()
                    .map(|ui| if w >= ui { 0.0 } else { NEG_INFINITY })
                    .collect();
                lpk
            })
            .collect();

        self.accum_score_and_integrate_asgn(
            logps,
            ncats,
            RowAssignAlg::Slice,
            &mut rng,
        );
    }

    fn accum_score_and_integrate_asgn(
        &mut self,
        mut logps: Vec<Vec<f64>>,
        ncats: usize,
        row_alg: RowAssignAlg,
        mut rng: &mut impl Rng,
    ) {
        for k in 0..ncats {
            for (_, ftr) in &self.ftrs {
                ftr.accum_score(&mut logps[k], k);
            }
        }

        let logps_t = transpose(&logps);
        let new_asgn_vec = match row_alg {
            RowAssignAlg::Slice => massflip_slice(logps_t, &mut rng),
            _ => massflip(logps_t, &mut rng),
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
        let dir = Dirichlet::new(dirvec.clone()).unwrap();
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
    pub fn update_alpha(&mut self, mut rng: &mut impl Rng) {
        self.asgn
            .update_alpha(braid_consts::MH_PRIOR_ITERS, &mut rng);
    }

    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
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
    fn observe_row(&mut self, row_ix: usize, k: usize) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.observe_datum(row_ix, k));
    }

    /// Have the components `k` forgets the data in `row_ix`
    fn forget_row(&mut self, row_ix: usize, k: usize) {
        self.ftrs
            .values_mut()
            .for_each(|ftr| ftr.forget_datum(row_ix, k));
    }

    /// Recompute the sufficient statistics in each component
    pub fn refresh_suffstats(&mut self, mut rng: &mut impl Rng) {
        for ftr in self.ftrs.values_mut() {
            ftr.reassign(&self.asgn, &mut rng);
        }
    }

    /// Get the likelihood of the data in this view given the current assignment
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
                .from_rng(&mut rng)
                .build()
                .unwrap();
            ViewBuilder::from_assignment(asgn).with_features(ftrs)
        }
        .from_rng(&mut rng)
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

impl GewekeSummarize for View {
    fn geweke_summarize(
        &self,
        settings: &ViewGewekeSettings,
    ) -> BTreeMap<String, f64> {
        let mut summary: BTreeMap<String, f64> = BTreeMap::new();

        let do_row_asgn_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::RowAssignment);

        let do_alpha_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::Alpha);

        if do_row_asgn_transition {
            summary.insert(String::from("ncats"), self.ncats() as f64);
        }

        if do_alpha_transition {
            summary.insert(String::from("CRP alpha"), self.asgn.alpha);
        }

        let col_settings = ColumnGewekeSettings::new(
            self.asgn.clone(),
            settings.transitions.clone(),
        );

        for (_, ftr) in &self.ftrs {
            // TODO: add column id to map key
            let mut ftr_summary = {
                let id: usize = ftr.id();
                let summary = ftr.geweke_summarize(&col_settings);
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

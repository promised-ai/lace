use super::State;

use crate::feature::ColModel;
use crate::transition::StateTransition;

use rand::seq::SliceRandom;
use rand::Rng;

impl State {
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

        let update_process_params = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewPriorProcessParams);

        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut col_ixs: Vec<usize> = (0..self.n_cols()).collect();
        col_ixs.shuffle(rng);

        col_ixs.drain(..).for_each(|col_ix| {
            self.reassign_col_gibbs(col_ix, update_process_params, rng);
        })
    }

    pub fn reassign_col_gibbs<R: Rng>(
        &mut self,
        col_ix: usize,
        update_process_params: bool,
        rng: &mut R,
    ) -> f64 {
        let ftr = self.extract_ftr(col_ix);
        self.insert_feature(ftr, update_process_params, rng)
    }

    /// Extract a feature from its view, unassign it, and drop the view if it
    /// is a singleton.
    pub(crate) fn extract_ftr(&mut self, ix: usize) -> ColModel {
        let v = self.asgn().asgn[ix];
        let ct = self.asgn().counts[v];
        let ftr = self.views[v].remove_feature(ix).unwrap();
        if ct == 1 {
            self.drop_view(v);
        }
        self.asgn_mut().unassign(ix);
        ftr
    }
}

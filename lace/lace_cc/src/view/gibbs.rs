use super::View;

use crate::constrain::RowConstrainer;
use crate::feature::Feature;
use lace_stats::prior_process::PriorProcessT;
use lace_stats::rv::misc::ln_pflip;
use rand::seq::SliceRandom;
use rand::Rng;

impl View {
    // Remove the row for the purposes of MCMC without deleting its data.
    pub(crate) fn remove_row(&mut self, row_ix: usize) {
        let k = self.asgn().asgn[row_ix];
        let is_singleton = self.asgn().counts[k] == 1;
        self.forget_row(row_ix, k);
        self.asgn_mut().unassign(row_ix);

        if is_singleton {
            self.drop_component(k);
        }
    }

    pub(crate) fn reinsert_row(
        &mut self,
        row_ix: usize,
        constrainer: &impl RowConstrainer,
        mut rng: &mut impl Rng,
    ) {
        let k_new = if self.asgn().n_cats == 0 {
            // If empty, assign to category zero
            debug_assert!(self.ftrs.values().all(|f| f.k() == 0));
            self.append_empty_component(&mut rng);
            0
        } else {
            // If not empty, do a Gibbs step
            let mut logps: Vec<f64> =
                Vec::with_capacity(self.asgn().n_cats + 1);

            self.asgn().counts.iter().enumerate().for_each(|(k, &ct)| {
                let w = self.prior_process.process.ln_gibbs_weight(ct);
                let ln_p_x = self.predictive_score_at(row_ix, k);
                let ln_constraint = constrainer.ln_constraint(row_ix, k);
                logps.push(w + ln_p_x + ln_constraint);
            });

            logps.push(
                self.prior_process
                    .process
                    .ln_singleton_weight(self.n_cats())
                    + self.singleton_score(row_ix)
                    + constrainer.ln_constraint(row_ix, self.n_cats()),
            );

            let k_new = ln_pflip(&logps, 1, false, &mut rng)[0];

            if k_new == self.n_cats() {
                self.append_empty_component(&mut rng);
            }

            k_new
        };

        self.observe_row(row_ix, k_new);
        self.asgn_mut().reassign(row_ix, k_new);
    }

    pub fn reassign_row_gibbs(
        &mut self,
        row_ix: usize,
        constrainer: &impl RowConstrainer,
        mut rng: &mut impl Rng,
    ) {
        self.remove_row(row_ix);
        self.reinsert_row(row_ix, constrainer, &mut rng);
    }

    /// Use the standard Gibbs kernel to reassign the rows
    pub fn reassign_rows_gibbs(&mut self, mut rng: &mut impl Rng) {
        let n_rows = self.n_rows();

        // The algorithm is not valid if the columns are not scanned in
        // random order
        let mut row_ixs: Vec<usize> = (0..n_rows).collect();
        row_ixs.shuffle(&mut rng);

        for row_ix in row_ixs {
            self.reassign_row_gibbs(row_ix, &(), &mut rng);
        }

        // NOTE: The oracle functions use the weights to compute probabilities.
        // Since the Gibbs algorithm uses implicit weights from the partition,
        // it does not explicitly update the weights. Non-updated weights means
        // wrong probabilities. To avoid this, we set the weights by the
        // partition here.
        self.weights = self.prior_process.weight_vec(false);
        debug_assert!(self.asgn().validate().is_valid());
    }
}

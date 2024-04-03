use super::State;

use crate::feature::Feature;
use crate::transition::StateTransition;

use lace_stats::prior_process::PriorProcessT;
use lace_stats::rv::misc::ln_pflip;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelDrainRange;
use rayon::iter::ParallelIterator;

impl State {
    // FIXME: NOT WORKING PROPERLY!
    /// Gibbs column transition where column transition probabilities are pre-
    /// computed in parallel
    pub fn reassign_cols_gibbs_parallel<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        mut rng: &mut R,
    ) {
        if self.n_cols() == 1 {
            return;
        }

        // Check if we're drawing view alpha. If not, we use the user-specified
        // alpha value for all temporary, singleton assignments
        let draw_process_params = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewPriorProcessParams);

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
                .map(|_| Xoshiro256Plus::from_rng(&mut rng).unwrap())
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
                        (0..m).map(|_| t_rng.gen()).collect();

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

                let mut v_new = ln_pflip(&logps, 1, false, rng)[0];

                if v_new != this_view_ix {
                    if v_new >= n_views {
                        // Moved to a singleton
                        let seed_ix = v_new - n_views;
                        let seed = seeds[seed_ix];

                        let prior_process =
                            self.create_tmp_assign(draw_process_params, seed);

                        let new_view =
                            crate::view::Builder::from_prior_process(
                                prior_process,
                            )
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
                            // desroyed singleton will be removed in
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
}

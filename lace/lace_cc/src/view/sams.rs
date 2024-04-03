use super::View;

use lace_stats::assignment::Assignment;
use lace_stats::prior_process;
use lace_utils::logaddexp;
use rand::seq::SliceRandom;
use rand::Rng;

impl View {
    /// Sequential adaptive merge-split (SAMS) row reassignment kernel
    pub fn reassign_rows_sams<R: Rng>(&mut self, rng: &mut R) {
        use rand::seq::IteratorRandom;

        let (i, j, zi, zj) = {
            let ixs = (0..self.n_rows()).choose_multiple(rng, 2);
            let i = ixs[0];
            let j = ixs[1];

            let zi = self.asgn().asgn[i];
            let zj = self.asgn().asgn[j];

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
        debug_assert!(self.asgn().validate().is_valid());
    }

    fn sams_split<R: Rng>(&mut self, i: usize, j: usize, rng: &mut R) {
        let zi = self.asgn().asgn[i];

        // FIXME: only works for CRP
        let logp_mrg =
            self.logm(zi) + self.prior_process.ln_f_partition(self.asgn());
        let (logp_spt, logq_spt, asgn_opt) =
            self.propose_split(i, j, false, rng);

        let asgn = asgn_opt.unwrap();

        if rng.gen::<f64>().ln() < logp_spt - logp_mrg - logq_spt {
            self.set_asgn(asgn, rng)
        }
    }

    fn sams_merge<R: Rng>(&mut self, i: usize, j: usize, rng: &mut R) {
        use std::cmp::Ordering;

        let zi = self.asgn().asgn[i];
        let zj = self.asgn().asgn[j];

        let (logp_spt, logq_spt, ..) = self.propose_split(i, j, true, rng);

        let asgn = {
            let zs = self
                .asgn()
                .asgn
                .iter()
                .map(|&z| match z.cmp(&zj) {
                    Ordering::Equal => zi,
                    Ordering::Greater => z - 1,
                    Ordering::Less => z,
                })
                .collect();

            prior_process::Builder::from_vec(zs)
                .with_process(self.prior_process.process.clone())
                .seed_from_rng(rng)
                .build()
                .unwrap()
                .asgn
        };

        self.append_empty_component(rng);
        asgn.asgn.iter().enumerate().for_each(|(ix, &z)| {
            if z == zi {
                self.force_observe_row(ix, self.n_cats());
            }
        });

        let logp_mrg =
            self.logm(self.n_cats()) + self.prior_process.ln_f_partition(&asgn);

        self.drop_component(self.n_cats());

        if rng.gen::<f64>().ln() < logp_mrg - logp_spt + logq_spt {
            self.set_asgn(asgn, rng)
        }
    }

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
            self.asgn()
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
                .asgn()
                .asgn
                .iter()
                .enumerate()
                .filter_map(|(ix, &z)| if z == zi { Some(ix) } else { None })
                .collect();

            row_ixs.shuffle(rng);
            row_ixs
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
        let zi = self.asgn().asgn[i];
        let zj = self.asgn().asgn[j];

        self.append_empty_component(rng);
        self.append_empty_component(rng);

        let zi_tmp = self.n_cats();
        let zj_tmp = zi_tmp + 1;

        self.force_observe_row(i, zi_tmp);
        self.force_observe_row(j, zj_tmp);

        let mut tmp_z: Vec<usize> = {
            // mark everything assigned to the split cluster as unassigned (-1)
            let mut zs: Vec<usize> = self
                .asgn()
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
                    self.asgn().asgn[ix] == zi
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
            logp += self.prior_process.ln_f_partition(self.asgn());
            None
        } else {
            tmp_z.iter_mut().for_each(|z| {
                if *z == zi_tmp {
                    *z = zi;
                } else if *z == zj_tmp {
                    *z = self.n_cats();
                }
            });

            // FIXME: create (draw) new process outside to carry forward alpha
            let asgn = prior_process::Builder::from_vec(tmp_z)
                .with_process(self.prior_process.process.clone())
                .seed_from_rng(rng)
                .build()
                .unwrap()
                .asgn;

            logp += self.prior_process.ln_f_partition(&asgn);
            Some(asgn)
        };

        // delete the last component twice since we appended two components
        self.drop_component(self.n_cats());
        self.drop_component(self.n_cats());

        (logp, logq, asgn)
    }
}

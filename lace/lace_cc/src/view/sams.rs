use crate::constrain::{RowSamsConstrainer, RowSamsInfo};

use super::View;

use lace_stats::assignment::Assignment;
use lace_stats::prior_process;
use lace_utils::logaddexp;
use rand::seq::SliceRandom;
use rand::Rng;

impl View {
    /// Sequential adaptive merge-split (SAMS) row reassignment kernel
    pub fn reassign_rows_sams<R: Rng>(
        &mut self,
        constrainer: &mut impl RowSamsConstrainer,
        rng: &mut R,
    ) {
        use rand::seq::IteratorRandom;

        let (i, j, z_i, z_j) = {
            let ixs = (0..self.n_rows()).choose_multiple(rng, 2);
            let i = ixs[0];
            let j = ixs[1];

            let z_i = self.asgn().asgn[i];
            let z_j = self.asgn().asgn[j];

            if z_i < z_j {
                (i, j, z_i, z_j)
            } else {
                (j, i, z_j, z_i)
            }
        };

        if z_i == z_j {
            constrainer.initialize(RowSamsInfo {
                i,
                j,
                z_i,
                z_j,
                z_split: self.asgn_mut().n_cats,
            });
            self.sams_split(i, j, constrainer, rng);
        } else {
            constrainer.initialize(RowSamsInfo {
                i,
                j,
                z_i,
                z_j,
                z_split: z_j,
            });
            assert!(z_i < z_j);
            self.sams_merge(i, j, constrainer, rng);
        }
        debug_assert!(self.asgn().validate().is_valid());
    }

    fn sams_split<R: Rng>(
        &mut self,
        i: usize,
        j: usize,
        constrainer: &mut impl RowSamsConstrainer,
        rng: &mut R,
    ) {
        let zi = self.asgn().asgn[i];

        let logp_mrg =
            self.logm(zi) + self.prior_process.ln_f_partition(self.asgn());
        let (logp_spt, logq_spt, asgn_opt) =
            self.propose_split(i, j, false, constrainer, rng);

        let asgn_prop = asgn_opt.unwrap();

        let ln_constraint = constrainer.ln_mh_constraint(&asgn_prop);

        if rng.gen::<f64>().ln()
            < logp_spt - logp_mrg - logq_spt + ln_constraint
        {
            self.set_asgn(asgn_prop, rng)
        }
    }

    fn sams_merge<R: Rng>(
        &mut self,
        i: usize,
        j: usize,
        constrainer: &mut impl RowSamsConstrainer,
        rng: &mut R,
    ) {
        use std::cmp::Ordering;

        let zi = self.asgn().asgn[i];
        let zj = self.asgn().asgn[j];

        let (logp_spt, logq_spt, ..) =
            self.propose_split(i, j, true, constrainer, rng);

        let asgn_prop = {
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
                .seed_from_rng(rng)
                .build()
                .unwrap()
                .asgn
        };

        self.append_empty_component(rng);
        asgn_prop.asgn.iter().enumerate().for_each(|(ix, &z)| {
            if z == zi {
                self.force_observe_row(ix, self.n_cats());
            }
        });

        let ln_constraint = constrainer.ln_mh_constraint(&asgn_prop);

        let logp_mrg = self.logm(self.n_cats())
            + self.prior_process.ln_f_partition(&asgn_prop);

        self.drop_component(self.n_cats());

        if rng.gen::<f64>().ln()
            < logp_mrg - logp_spt + logq_spt + ln_constraint
        {
            self.set_asgn(asgn_prop, rng)
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
        constrainer: &mut impl RowSamsConstrainer,
        rng: &mut R,
    ) -> (f64, f64, Option<Assignment>) {
        let zi = self.asgn().asgn[i];
        let zj = self.asgn().asgn[j];

        self.append_empty_component(rng);
        self.append_empty_component(rng);

        let z_i_tmp = self.n_cats();
        let z_j_tmp = z_i_tmp + 1;

        self.force_observe_row(i, z_i_tmp);
        self.force_observe_row(j, z_j_tmp);

        let mut tmp_z: Vec<usize> = {
            // mark everything assigned to the split cluster as unassigned (-1)
            let mut zs: Vec<usize> = self
                .asgn()
                .iter()
                .map(|&z| if z == zi { std::usize::MAX } else { z })
                .collect();
            zs[i] = z_i_tmp;
            zs[j] = z_j_tmp;
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
                let (ln_c_zi, ln_c_zj) = constrainer.ln_sis_contstraints(ix);
                let logp_z_i =
                    nk_i.ln() + self.predictive_score_at(ix, z_i_tmp) + ln_c_zi;
                let logp_z_j =
                    nk_j.ln() + self.predictive_score_at(ix, z_j_tmp) + ln_c_zj;
                let lognorm = logaddexp(logp_z_i, logp_z_j);

                let assign_to_zi = if calc_reverse {
                    self.asgn().asgn[ix] == zi
                } else {
                    rng.gen::<f64>().ln() < logp_z_i - lognorm
                };

                if assign_to_zi {
                    constrainer.sis_assign(ix, false);
                    logq += logp_z_i - lognorm;
                    self.force_observe_row(ix, z_i_tmp);
                    nk_i += 1.0;
                    tmp_z[ix] = z_i_tmp;
                } else {
                    constrainer.sis_assign(ix, true);
                    logq += logp_z_j - lognorm;
                    self.force_observe_row(ix, z_j_tmp);
                    nk_j += 1.0;
                    tmp_z[ix] = z_j_tmp;
                }
            });

        let mut logp = self.logm(z_i_tmp) + self.logm(z_j_tmp);

        let asgn = if calc_reverse {
            logp += self.prior_process.ln_f_partition(self.asgn());
            None
        } else {
            tmp_z.iter_mut().for_each(|z| {
                if *z == z_i_tmp {
                    *z = zi;
                } else if *z == z_j_tmp {
                    *z = self.n_cats();
                }
            });

            let asgn = prior_process::Builder::from_vec(tmp_z)
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

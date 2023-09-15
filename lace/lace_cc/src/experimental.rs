//! Experimental implementations
use lace_consts::rv::experimental::{Sb, Sbd, SbdSuffStat};
use lace_consts::rv::prelude::Categorical;
use lace_consts::rv::traits::{ConjugatePrior, Rv};
use lace_data::{Container, Datum, FeatureData, SparseContainer};
use lace_geweke::{GewekeModel, GewekeSummarize};
use lace_stats::experimental::sbd::SbdHyper;
use once_cell::sync::OnceCell;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::assignment::Assignment;
use crate::component::ConjugateComponent;
use crate::feature::geweke::{ColumnGewekeSettings, GewekeColumnSummary};
use crate::feature::{ColModel, Column, FType, Feature, TranslateDatum};
use crate::state::State;
use crate::traits::AccumScore;
use crate::traits::LacePrior;
use crate::view::View;
use lace_utils::{Matrix, Shape};

#[derive(Debug)]
pub struct ViewConstraintMatrix {
    pub col_ixs: Vec<usize>,
    pub matrix: lace_utils::Matrix<f64>,
}

impl ViewConstraintMatrix {
    pub fn zeros_like(other: &Self) -> Self {
        ViewConstraintMatrix {
            col_ixs: other.col_ixs.clone(),
            matrix: lace_utils::Matrix::from_raw_parts(
                vec![0.0; other.matrix.nelem()],
                other.matrix.n_rows(),
            ),
        }
    }

    pub fn n_cols(&self) -> usize {
        self.col_ixs.len()
    }

    pub fn n_rows(&self) -> usize {
        self.matrix.n_cols()
    }

    pub fn n_cats(&self) -> usize {
        self.matrix.n_rows()
    }
}

impl View {
    /// Create the slice matrix and prepare the view for the row reassignment
    ///
    /// # Warning
    /// The row reassignment must be completed or else this function will leave
    /// an invalid view.
    pub fn slice_matrix<R: Rng>(
        &mut self,
        targets: &HashSet<usize>,
        rng: &mut R,
    ) -> Matrix<f64> {
        let logps = self._slice_matrix(rng);
        self.accum_score(targets, logps)
    }

    // /// Use the improved slice algorithm to reassign the rows
    // pub fn reassign_rows_slice_constrained(
    //     &mut self,
    //     conditions: Option<&HashSet<usize>>,
    //     constrainer: &Matrix<f64>,
    //     mut rng: &mut impl Rng,
    // ) {
    //     let mut logps = self._slice_matrix(rng);

    //     assert_eq!(logps.shape(), constrainer.shape());

    //     logps
    //         .raw_values_mut()
    //         .iter_mut()
    //         .zip(constrainer.raw_values().iter())
    //         .for_each(|(a, b)| *a += b);

    //     let n_cats = logps.n_rows();

    //     if let Some(targets) = conditions {
    //         self.accum_score_and_integrate_asgn_cnd(
    //             targets,
    //             logps,
    //             n_cats,
    //             &crate::alg::RowAssignAlg::Slice,
    //             &mut rng,
    //         );
    //     } else {
    //         self.accum_score_and_integrate_asgn(
    //             logps,
    //             n_cats,
    //             &crate::alg::RowAssignAlg::Slice,
    //             &mut rng,
    //         );
    //     }
    // }

    pub fn integrate_slice_assign(
        &mut self,
        logps: ViewConstraintMatrix,
        rng: &mut impl Rng,
    ) {
        let logps = logps.matrix.implicit_transpose();
        let n_cats = logps.n_cols();
        let new_asgn_vec = crate::massflip::massflip_slice_mat_par(&logps, rng);

        self.integrate_finite_asgn(new_asgn_vec, n_cats, rng);
    }

    pub fn update_component<R: Rng>(&mut self, k: usize, rng: &mut R) {
        for ftr in self.ftrs.values_mut() {
            ftr.update_component(k, rng)
        }
    }

    /// Sequential adaptive merge-split (SAMS) row reassignment kernel
    pub fn reassign_rows_sams_constrained<R: Rng>(
        &mut self,
        constrainer: &Matrix<f64>,
        rng: &mut R,
    ) {
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
            self.sams_split_constrained(i, j, constrainer, rng);
        } else {
            assert!(zi < zj);
            self.sams_merge_constrained(i, j, constrainer, rng);
        }
    }

    fn sams_merge_constrained<R: Rng>(
        &mut self,
        i: usize,
        j: usize,
        constrainer: &Matrix<f64>,
        rng: &mut R,
    ) {
        use crate::assignment::lcrp;
        use crate::assignment::AssignmentBuilder;
        use std::cmp::Ordering;

        let zi = self.asgn.asgn[i];
        let zj = self.asgn.asgn[j];

        let (logp_spt, logq_spt, ..) =
            self.propose_split_constrained(i, j, true, constrainer, rng);

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
            + lcrp(asgn.len(), &asgn.counts, asgn.alpha)
            + self
                .asgn
                .asgn
                .iter()
                .enumerate()
                .filter(|(_, &z)| z == zi)
                .map(|(ix, &z)| constrainer[(zi, ix)])
                .sum::<f64>();

        self.drop_component(self.n_cats());

        if rng.gen::<f64>().ln() < logp_mrg - logp_spt + logq_spt {
            self.set_asgn(asgn, rng)
        }
    }

    fn sams_split_constrained<R: Rng>(
        &mut self,
        i: usize,
        j: usize,
        constrainer: &Matrix<f64>,
        rng: &mut R,
    ) {
        use crate::assignment::lcrp;

        let zi = self.asgn.asgn[i];

        let logp_mrg = self.logm(zi)
            + lcrp(self.asgn.len(), &self.asgn.counts, self.asgn.alpha)
            + self
                .asgn
                .asgn
                .iter()
                .enumerate()
                .filter(|(_, &z)| z == zi)
                .map(|(ix, &z)| constrainer[(z, ix)])
                .sum::<f64>();
        let (logp_spt, logq_spt, asgn_opt) =
            self.propose_split_constrained(i, j, false, constrainer, rng);

        let asgn = asgn_opt.unwrap();

        if rng.gen::<f64>().ln() < logp_spt - logp_mrg - logq_spt {
            self.set_asgn(asgn, rng)
        }
    }

    fn propose_split_constrained<R: Rng>(
        &mut self,
        i: usize,
        j: usize,
        calc_reverse: bool,
        constrainer: &Matrix<f64>,
        rng: &mut R,
    ) -> (f64, f64, Option<Assignment>) {
        use crate::assignment::lcrp;
        use crate::assignment::AssignmentBuilder;
        use lace_utils::logaddexp;

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

        let mut logp = 0.0;

        row_ixs
            .iter()
            .filter(|&&ix| !(ix == i || ix == j))
            .for_each(|&ix| {
                let logp_ci = constrainer[(zi, ix)];
                let logp_cj = constrainer[(zi_tmp, ix)];
                let logp_zi =
                    nk_i.ln() + self.predictive_score_at(ix, zi_tmp) + logp_ci;
                let logp_zj =
                    nk_j.ln() + self.predictive_score_at(ix, zj_tmp) + logp_cj;
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
                    logp += logp_ci;
                } else {
                    logq += logp_zj - lognorm;
                    self.force_observe_row(ix, zj_tmp);
                    nk_j += 1.0;
                    tmp_z[ix] = zj_tmp;
                    logp += logp_cj;
                }
            });

        logp += self.logm(zi_tmp) + self.logm(zj_tmp);

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
}

impl State {
    /// delete any components extend beyond the assignment
    pub fn drop_extra_components(&mut self) {
        for view in self.views.iter_mut() {
            let n_cats = view.asgn.n_cats;
            for ftr in view.ftrs.values_mut() {
                (n_cats..ftr.k()).for_each(|_| ftr.drop_component(n_cats));
            }
        }
    }

    /// Creates logp matrices and prepares the state for row-reassignment via
    /// slice sampling.
    ///
    /// # Warning
    /// This function will result in an invalids state if the row reassignment
    /// is not completed using the `slice_reassign_rows_with` method
    pub fn slice_view_matrices<R: Rng>(
        &mut self,
        targets: &HashSet<usize>,
        rng: &mut R,
    ) -> Vec<ViewConstraintMatrix> {
        let seeds =
            (0..self.n_views()).map(|_| rng.gen()).collect::<Vec<u64>>();

        self.views
            .par_iter_mut()
            .zip(seeds.par_iter())
            .map(|(view, seed)| {
                let mut trng = Xoshiro256Plus::seed_from_u64(*seed);
                let matrix = view.slice_matrix(targets, &mut trng);
                let col_ixs = view.ftrs.keys().cloned().collect();
                ViewConstraintMatrix { matrix, col_ixs }
            })
            .collect()
    }

    /// Complete the slice row reassignment with pre-computed logp matrices
    pub fn integrate_slice_assign<R: Rng>(
        &mut self,
        mut view_matricies: Vec<ViewConstraintMatrix>,
        rng: &mut R,
    ) {
        let mut seeds: Vec<u64> =
            (0..self.n_views()).map(|_| rng.gen()).collect();

        self.views
            .par_iter_mut()
            .zip_eq(view_matricies.par_drain(..))
            .zip_eq(seeds.par_drain(..))
            .for_each(|((view, logps), seed)| {
                let mut rng = Xoshiro256Plus::seed_from_u64(seed);
                view.integrate_slice_assign(logps, &mut rng);
            });
    }

    pub fn reassign_rows_sams_constrained<R: Rng>(
        &mut self,
        view_constraints: &[ViewConstraintMatrix],
        rng: &mut R,
    ) {
        use rand_xoshiro::Xoroshiro128Plus;
        let mut t_rngs = (0..self.n_views())
            .map(|_| Xoroshiro128Plus::seed_from_u64(rng.gen()))
            .collect::<Vec<_>>();
        self.views
            .par_iter_mut()
            .zip_eq(view_constraints.par_iter())
            .zip_eq(t_rngs.par_iter_mut())
            .for_each(|((view, constraints), t_rng)| {
                view.reassign_rows_sams_constrained(&constraints.matrix, t_rng);
            })
    }

    // append a linking partition column to the table
    pub fn append_partition_column<R: Rng>(
        &mut self,
        asgn: &Assignment,
        rng: &mut R,
    ) {
        // 1. choose a random view for this column
        let view_ix = rng.gen_range(0..self.n_views());

        // 2. Initialize the ColModel
        let ftr: ColModel = {
            let n_cols = self.n_cols();
            let n_cats = self.views[view_ix].n_cats();

            let hyper = SbdHyper::default();
            let prior = hyper.draw(asgn.n_cats, rng);
            let components = (0..n_cats)
                .map(|_| ConjugateComponent::new(prior.draw(rng)))
                .collect::<Vec<_>>();

            ColModel::Index(Column {
                id: n_cols,
                data: SparseContainer::from(asgn.asgn.clone()),
                components,
                prior,
                hyper,
                ignore_hyper: false,
                ln_m_cache: OnceCell::new(),
            })
        };

        // 3. Insert into the view
        self.views[view_ix].insert_feature(ftr, rng);

        // 4. increment assignment
        self.asgn.asgn.push(view_ix);
        self.asgn.counts[view_ix] += 1;
    }
}

// impl AccumScore<usize> for Dpd {}
impl AccumScore<usize> for Sbd {}

// impl LacePrior<usize, Dpd, DpdHyper> for DpdPrior {
//     fn empty_suffstat(&self) -> DpdSuffStat {
//         DpdSuffStat::new()
//     }

//     fn invalid_temp_component(&self) -> Dpd {
//         Dpd::uniform(self.k(), self.m())
//     }

//     fn score_column<I: Iterator<Item = DpdSuffStat>>(&self, stats: I) -> f64 {
//         use lace_stats::rv::data::DataOrSuffStat;
//         // let cache = self.ln_m_cache();
//         stats
//             .map(|stat| {
//                 let x = DataOrSuffStat::SuffStat(&stat);
//                 self.ln_m_with_cache(&(), &x)
//             })
//             .sum::<f64>()
//     }
// }
//
// impl TranslateDatum<usize> for Column<usize, Dpd, DpdPrior, DpdHyper> {
//     fn translate_datum(datum: Datum) -> usize {
//         match datum {
//             Datum::Index(x) => x,
//             _ => panic!("Invalid Datum variant for conversion"),
//         }
//     }

//     fn translate_value(x: usize) -> Datum {
//         Datum::Index(x)
//     }

//     fn translate_feature_data(data: FeatureData) -> SparseContainer<usize> {
//         match data {
//             FeatureData::Index(xs) => xs,
//             _ => panic!("Invalid FeatureData variant for conversion"),
//         }
//     }

//     fn translate_container(xs: SparseContainer<usize>) -> FeatureData {
//         FeatureData::Index(xs)
//     }

//     fn ftype() -> FType {
//         FType::Index
//     }
// }

impl LacePrior<usize, Sbd, SbdHyper> for Sb {
    fn empty_suffstat(&self) -> SbdSuffStat {
        SbdSuffStat::new()
    }

    fn invalid_temp_component(&self) -> Sbd {
        Sbd::new(1.0, Some(1337)).unwrap()
    }

    fn score_column<I: Iterator<Item = SbdSuffStat>>(&self, stats: I) -> f64 {
        use lace_stats::rv::data::DataOrSuffStat;
        // let cache = self.ln_m_cache();
        stats
            .map(|stat| {
                let x = DataOrSuffStat::SuffStat(&stat);
                self.ln_m_with_cache(&(), &x)
            })
            .sum::<f64>()
        // unimplemented!()
    }
}

impl TranslateDatum<usize> for Column<usize, Sbd, Sb, SbdHyper> {
    fn translate_datum(datum: Datum) -> usize {
        match datum {
            Datum::Index(x) => x,
            _ => panic!("Invalid Datum variant for conversion"),
        }
    }

    fn translate_value(x: usize) -> Datum {
        Datum::Index(x)
    }

    fn translate_feature_data(data: FeatureData) -> SparseContainer<usize> {
        match data {
            FeatureData::Index(xs) => xs,
            _ => panic!("Invalid FeatureData variant for conversion"),
        }
    }

    fn translate_container(xs: SparseContainer<usize>) -> FeatureData {
        FeatureData::Index(xs)
    }

    fn ftype() -> FType {
        FType::Index
    }
}

// Geweke for Sbd
// --------------
impl GewekeModel for Column<usize, Sbd, Sb, SbdHyper> {
    #[must_use]
    fn geweke_from_prior(
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) -> Self {
        let k = 5;

        let f = Categorical::uniform(k);
        let xs: Vec<usize> = f.sample(settings.asgn().len(), &mut rng);

        let data = SparseContainer::from(xs); // initial data is resampled anyway
        let hyper = SbdHyper::default();
        let prior = if settings.fixed_prior() {
            Sb::new(1.0, k, Some(rng.gen()))
        } else {
            hyper.draw(k, &mut rng)
        };
        let mut col = Column::new(0, data, prior, hyper);
        col.init_components(settings.asgn().n_cats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(
        &mut self,
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) {
        self.update_components(&mut rng);
        if !settings.fixed_prior() {
            self.update_prior_params(&mut rng);
        }
    }
}

impl GewekeSummarize for Column<usize, Sbd, Sb, SbdHyper> {
    type Summary = GewekeColumnSummary;

    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> Self::Summary {
        let x_sum = self.data.present_cloned().iter().sum::<usize>();

        fn sum_sq(logws: &[f64]) -> f64 {
            logws.iter().fold(0.0, |acc, lw| {
                let lw_exp = lw.exp();
                lw_exp.mul_add(lw_exp, acc)
            })
        }

        let kf = self.components.len() as f64;

        let ln_weights: Vec<Vec<f64>> = self
            .components
            .iter()
            .map(|cpnt| {
                cpnt.fx
                    .inner
                    .read()
                    .map(|inner| inner.ln_weights.clone())
                    .unwrap()
            })
            .collect();

        let sq_weight_mean: f64 =
            ln_weights.iter().map(|lnws| sum_sq(lnws)).sum::<f64>() / kf;

        let weight_mean: f64 = ln_weights
            .iter()
            .map(|lnws| {
                let kw = lnws.len() as f64;
                lnws.iter().sum::<f64>() / kw
            })
            .sum::<f64>()
            / kf;

        GewekeColumnSummary::Categorical {
            x_sum,
            sq_weight_mean,
            weight_mean,
            prior_alpha: if !settings.fixed_prior() {
                Some(self.prior.alpha())
            } else {
                None
            },
        }
    }
}

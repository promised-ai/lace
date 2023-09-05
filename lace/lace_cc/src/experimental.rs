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
pub struct ViewSliceMatrix {
    pub col_ixs: Vec<usize>,
    pub matrix: lace_utils::Matrix<f64>,
}

impl ViewSliceMatrix {
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
    #[cfg(feature = "experimental")]
    pub fn slice_matrix<R: Rng>(
        &mut self,
        targets: &HashSet<usize>,
        rng: &mut R,
    ) -> Matrix<f64> {
        let logps = self._slice_matrix(rng);
        self.accum_score(targets, logps)
    }

    pub fn integrate_slice_assign(
        &mut self,
        logps: ViewSliceMatrix,
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

    // pub fn increment_parent_slice_matrix(
    //     &self,
    //     mut matrix: ViewSliceMatrix,
    // ) -> Result<ViewSliceMatrix, CabnError> {
    //     for col_ix in matrix.col_ixs {
    //         let ftr = self.ftrs[&col_ix];
    //         for k in 0..matrix.n_cats() {
    //             let x = Datum::Categorical(k);
    //             for row_ix in 0..self.n_rows() {
    //                 ftr.
    //             }
    //         }
    //     }
    //     matrix
    // }
}

impl State {
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
    ) -> Vec<ViewSliceMatrix> {
        let seeds =
            (0..self.n_views()).map(|_| rng.gen()).collect::<Vec<u64>>();

        self.views
            .par_iter_mut()
            .zip(seeds.par_iter())
            .map(|(view, seed)| {
                let mut trng = Xoshiro256Plus::seed_from_u64(*seed);
                let matrix = view.slice_matrix(targets, &mut trng);
                let col_ixs = view.ftrs.keys().cloned().collect();
                ViewSliceMatrix { matrix, col_ixs }
            })
            .collect()
    }

    /// Complete the slice row reassignment with pre-computed logp matrices
    pub fn integrate_slice_assign<R: Rng>(
        &mut self,
        mut view_matricies: Vec<ViewSliceMatrix>,
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

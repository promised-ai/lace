use super::View;

use crate::alg::RowAssignAlg;
use crate::constrain::RowConstrainer;
use crate::feature::Feature;

use lace_stats::prior_process::PriorProcessT;
use lace_stats::rv::dist::Dirichlet;
use lace_stats::rv::traits::Rv;
use lace_utils::Matrix;
use lace_utils::Shape;
use rand::Rng;

impl View {
    /// Use the improved slice algorithm to reassign the rows
    pub fn reassign_rows_slice(
        &mut self,
        constrainer: &impl RowConstrainer,
        mut rng: &mut impl Rng,
    ) {
        self.resample_weights(false, &mut rng);

        let weights: Vec<f64> = {
            // FIXME: only works for dirichlet
            let dirvec = self.prior_process.weight_vec_unnormed(true);
            let dir = Dirichlet::new(dirvec).unwrap();
            dir.draw(&mut rng)
        };

        let us: Vec<f64> = self
            .asgn()
            .asgn
            .iter()
            .map(|&zi| {
                let wi: f64 = weights[zi];
                let u: f64 = rng.gen::<f64>();
                u * wi
            })
            .collect();

        let u_star: f64 =
            us.iter()
                .fold(1.0, |umin, &ui| if ui < umin { ui } else { umin });

        let weights = self
            .prior_process
            .process
            .slice_sb_extend(weights, u_star, &mut rng);

        let n_new_cats = weights.len() - self.weights.len();
        let n_cats = weights.len();

        for _ in 0..n_new_cats {
            self.append_empty_component(&mut rng);
        }

        // initialize truncated log probabilities
        let logps = {
            let mut values = Vec::with_capacity(weights.len() * self.n_rows());
            weights.iter().for_each(|w| {
                us.iter().for_each(|ui| {
                    let value =
                        if w >= ui { 0.0 } else { std::f64::NEG_INFINITY };
                    values.push(value);
                });
            });
            let matrix = Matrix::from_raw_parts(values, n_cats);
            debug_assert_eq!(matrix.n_cols(), us.len());
            debug_assert_eq!(matrix.n_rows(), weights.len());
            matrix
        };

        self.accum_score_and_integrate_asgn(
            logps,
            n_cats,
            RowAssignAlg::Slice,
            constrainer,
            &mut rng,
        );
    }

    pub(crate) fn accum_score_and_integrate_asgn(
        &mut self,
        mut logps: Matrix<f64>,
        n_cats: usize,
        row_alg: RowAssignAlg,
        constrainer: &impl RowConstrainer,
        rng: &mut impl Rng,
    ) {
        use rayon::prelude::*;

        logps.par_rows_mut().enumerate().for_each(|(k, logp)| {
            self.ftrs.values().for_each(|ftr| {
                ftr.accum_score(logp, k);
                logp.iter_mut().enumerate().for_each(|(row_ix, p)| {
                    *p += constrainer.ln_constraint(row_ix, k);
                })
            })
        });

        // Implicit transpose does not change the memory layout, just the
        // indexing.
        let logps = logps.implicit_transpose();
        debug_assert_eq!(logps.n_rows(), self.n_rows());

        let new_asgn_vec = match row_alg {
            RowAssignAlg::Slice => {
                crate::massflip::massflip_slice_mat_par(&logps, rng)
            }
            _ => crate::massflip::massflip(&logps, rng),
        };

        self.integrate_finite_asgn(new_asgn_vec, n_cats, rng);
    }
}

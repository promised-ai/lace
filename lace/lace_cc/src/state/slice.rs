use crate::state::State;

use crate::feature::ColModel;
use crate::feature::Feature;
use crate::transition::StateTransition;

use lace_stats::prior_process::PriorProcessT;
use lace_stats::rv::dist::Dirichlet;
use lace_stats::rv::traits::Sampleable;
use lace_utils::unused_components;
use lace_utils::Matrix;
use rand::Rng;

use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

impl State {
    /// Reassign columns to views using the improved slice sampler
    pub fn reassign_cols_slice<R: Rng>(
        &mut self,
        transitions: &[StateTransition],
        rng: &mut R,
    ) {
        if self.n_cols() == 1 {
            return;
        }

        self.resample_weights(false, rng);

        let n_cols = self.n_cols();

        let weights: Vec<f64> = {
            let dirvec = self.prior_process.weight_vec_unnormed(true);
            // FIXME: this only works for Dirichlet process!
            let dir = Dirichlet::new(dirvec).unwrap();
            dir.draw(rng)
        };

        let us: Vec<f64> = self
            .asgn()
            .asgn
            .iter()
            .map(|&zi| {
                let wi: f64 = weights[zi];
                let u: f64 = rng.gen();
                u * wi
            })
            .collect();

        let u_star: f64 =
            us.iter()
                .fold(1.0, |umin, &ui| if ui < umin { ui } else { umin });

        // Variable shadowing
        let weights = self
            .prior_process
            .process
            .slice_sb_extend(weights, u_star, rng);

        let n_new_views = weights.len() - self.weights.len();
        let n_views = weights.len();

        let mut ftrs: Vec<ColModel> = Vec::with_capacity(n_cols);
        for (i, &v) in self.prior_process.asgn.iter().enumerate() {
            ftrs.push(
                self.views[v].remove_feature(i).expect("Feature missing"),
            );
        }

        let draw_alpha = transitions
            .iter()
            .any(|&t| t == StateTransition::ViewPriorProcessParams);
        for _ in 0..n_new_views {
            self.append_empty_view(draw_alpha, rng);
        }

        // initialize truncated log probabilities
        let logps = {
            let values: Vec<f64> = ftrs
                .par_iter()
                .zip(us.par_iter())
                .flat_map(|(ftr, ui)| {
                    self.views
                        .iter()
                        .zip(weights.iter())
                        .map(|(view, w)| {
                            if w >= ui {
                                ftr.asgn_score(view.asgn())
                            } else {
                                std::f64::NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();

            Matrix::from_raw_parts(values, ftrs.len())
        };

        let new_asgn_vec = crate::massflip::massflip_slice_mat_par(&logps, rng);

        self.integrate_finite_asgn(new_asgn_vec, ftrs, n_views, rng);
        self.resample_weights(false, rng);
    }

    pub(crate) fn integrate_finite_asgn<R: Rng>(
        &mut self,
        mut new_asgn_vec: Vec<usize>,
        mut ftrs: Vec<ColModel>,
        n_views: usize,
        rng: &mut R,
    ) {
        let unused_views = unused_components(n_views, &new_asgn_vec);

        for v in unused_views {
            self.drop_view(v);
            for z in new_asgn_vec.iter_mut() {
                if *z > v {
                    *z -= 1
                };
            }
        }

        self.asgn_mut()
            .set_asgn(new_asgn_vec)
            .expect("new_asgn_vec is invalid");

        for (ftr, &v) in ftrs.drain(..).zip(self.prior_process.asgn.asgn.iter())
        {
            self.views[v].insert_feature(ftr, rng)
        }
    }
}

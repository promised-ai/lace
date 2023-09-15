use super::Engine;
use crate::cc::experimental::ViewConstraintMatrix;
use lace_codebook::ColMetadata;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashSet;

use rayon::prelude::*;

#[derive(Debug)]
pub struct ConstraintMatrices {
    // state_matrices[s][v] is the logp matrix for view v under state s
    pub state_matrices: Vec<Vec<ViewConstraintMatrix>>,
}

impl ConstraintMatrices {
    /// Create ConstraintMatrices the same shape as `self` but fully populated
    /// with zeros
    pub fn zeros_like(other: &Self) -> Self {
        Self {
            state_matrices: other
                .state_matrices
                .iter()
                .map(|view_mats| {
                    view_mats
                        .iter()
                        .map(|mat| ViewConstraintMatrix::zeros_like(mat))
                        .collect()
                })
                .collect(),
        }
    }

    /// Add the entries of the equally-sized ConstraintMatrices `other` to
    /// `self`
    pub fn recieve_constraints(&mut self, other: &Self) {
        self.state_matrices
            .par_iter_mut()
            .zip_eq(other.state_matrices.par_iter())
            .for_each(|(view_mats_s, view_mats_o)| {
                view_mats_s
                    .par_iter_mut()
                    .zip_eq(view_mats_o.par_iter())
                    .for_each(|(mat_s, mat_o)| {
                        mat_s
                            .matrix
                            .raw_values_mut()
                            .iter_mut()
                            .zip(mat_o.matrix.raw_values().iter())
                            .for_each(|(p_s, p_o)| {
                                *p_s += p_o;
                            })
                    })
            })
    }
}

// experimental impls
impl Engine {
    pub fn slice_row_matrices(
        &mut self,
        targets: &HashSet<usize>,
    ) -> ConstraintMatrices {
        let mut rng = self.rng.clone();
        let seeds: Vec<u64> = (0..self.n_states()).map(|_| rng.gen()).collect();

        let state_matrices: Vec<Vec<ViewConstraintMatrix>> = self
            .states
            .par_iter_mut()
            .zip(seeds.par_iter())
            .map(|(state, seed)| {
                let mut rng = Xoshiro256Plus::seed_from_u64(*seed);
                state.slice_view_matrices(targets, &mut rng)
            })
            .collect();

        ConstraintMatrices { state_matrices }
    }

    pub fn integrate_slice_assign(&mut self, mut matrices: ConstraintMatrices) {
        let seeds: Vec<u64> =
            (0..self.n_states()).map(|_| self.rng.gen()).collect();

        self.states
            .par_iter_mut()
            .zip_eq(matrices.state_matrices.par_drain(..))
            .zip_eq(seeds.par_iter())
            .for_each(|((state, view_matrices), seed)| {
                let mut rng = Xoshiro256Plus::seed_from_u64(*seed);
                state.integrate_slice_assign(view_matrices, &mut rng)
            })
    }

    pub fn reassign_rows_sams_constrained(
        &mut self,
        constraints: &ConstraintMatrices,
    ) {
        use rand_xoshiro::Xoroshiro128Plus;
        let mut t_rngs = (0..self.n_states())
            .map(|_| Xoroshiro128Plus::seed_from_u64(self.rng.gen()))
            .collect::<Vec<_>>();
        self.states
            .par_iter_mut()
            .zip_eq(constraints.state_matrices.par_iter())
            .zip_eq(t_rngs.par_iter_mut())
            .for_each(|((state, constraints), t_rng)| {
                state.reassign_rows_sams_constrained(&constraints, t_rng);
            })
    }

    pub fn connect(&mut self, parent: &Engine) {
        use crate::HasStates;

        // To be compatible, the engines must have the same number of states and
        // the same number of rows
        assert_eq!(parent.n_states(), self.n_states());
        assert_eq!(parent.n_rows(), self.n_rows());
        let last_col = self.n_cols();

        self.states.iter_mut().zip(parent.states.iter()).for_each(
            |(child, parent)| {
                // For now, assert flat column structure
                assert_eq!(parent.n_views(), 1);
                assert_eq!(child.n_views(), 1);
                child.append_partition_column(
                    &parent.views[0].asgn,
                    &mut self.rng,
                );
            },
        );
        let col_md = ColMetadata {
            name: format!("_PARTITION_{last_col}"),
            coltype: lace_codebook::ColType::Index {
                k: 10,
                hyper: None,
                prior: None,
            },
            notes: None,
            missing_not_at_random: false,
            latent: false,
        };
        self.codebook.col_metadata.push(col_md).unwrap();
    }
}

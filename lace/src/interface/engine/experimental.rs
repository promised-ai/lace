use super::Engine;
use crate::cc::experimental::ViewSliceMatrix;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashSet;

use rayon::prelude::*;

pub struct SliceRowMatrices {
    // state_matrices[s][v] is the logp matric for view v under state s
    pub state_matrices: Vec<Vec<ViewSliceMatrix>>,
}

// experimental impls
impl Engine {
    pub fn slice_row_matrices(
        &mut self,
        targets: &HashSet<usize>,
    ) -> SliceRowMatrices {
        let seeds: Vec<u64> =
            (0..self.n_states()).map(|_| self.rng.gen()).collect();

        let state_matrices: Vec<Vec<ViewSliceMatrix>> = self
            .states
            .par_iter_mut()
            .zip(seeds.par_iter())
            .map(|(state, seed)| {
                let mut rng = Xoshiro256Plus::seed_from_u64(*seed);
                state.slice_view_matrices(targets, &mut rng)
            })
            .collect();

        SliceRowMatrices { state_matrices }
    }

    pub fn integrate_slice_assign(&mut self, mut matrices: SliceRowMatrices) {
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
}

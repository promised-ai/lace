use braid_data::AccumScore as _;
use braid_data::SparseContainer;
use braid_stats::labeler::{Label, Labeler};

// use rayon::prelude::*;
use rv::data::CategoricalDatum;
use rv::dist::{Categorical, Gaussian, Poisson};
use rv::traits::Rv;

/// Score accumulation for `finite_cpu` and `slice` row transition kernels.
///
/// Provides two functions to add the scores (log likelihood) of a vector of
/// data to a vector of existing scores.
pub trait AccumScore<X: Clone + Default>: Rv<X> + Sync {
    // XXX: Default implementations can be improved upon by pre-computing
    // normalizers
    fn accum_score(&self, scores: &mut [f64], container: &SparseContainer<X>) {
        container.accum_score(scores, &|x| self.ln_f(x))
    }
}

impl<X: CategoricalDatum + Default> AccumScore<X> for Categorical {}
impl AccumScore<Label> for Labeler {}
impl AccumScore<u32> for Poisson {}
impl AccumScore<f64> for Gaussian {}

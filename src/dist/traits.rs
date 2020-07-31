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

// Since we don't care about the scores being normalized properly we can save
// some computation by not normalizing.
// impl AccumScore<f64> for Gaussian {
//     fn accum_score_par(
//         &self,
//         scores: &mut [f64],
//         xs: &[f64],
//         present: &[bool],
//     ) {
//         let mu = self.mu();
//         let sigma = self.sigma();
//         let log_z = sigma.ln() + rv::consts::HALF_LN_2PI;

//         let xs_iter = xs.par_iter().zip_eq(present.par_iter());
//         scores
//             .par_iter_mut()
//             .zip_eq(xs_iter)
//             .for_each(|(score, (x, &r))| {
//                 if r {
//                     let term = (x - mu) / sigma;
//                     let loglike = -0.5 * term * term - log_z;
//                     *score += loglike;
//                 }
//             });
//     }
// }

impl<X: CategoricalDatum + Default> AccumScore<X> for Categorical {}
impl AccumScore<Label> for Labeler {}
impl AccumScore<u32> for Poisson {}
impl AccumScore<f64> for Gaussian {}

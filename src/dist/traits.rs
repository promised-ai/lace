use braid_stats::labeler::{Label, Labeler};
use rayon::prelude::*;
use rv::data::CategoricalDatum;
use rv::dist::{Categorical, Gaussian};
use rv::traits::Rv;

/// Score accumulation for `finite_cpu` and `slice` row transition kernels.
///
/// Provides two functions to add the scores (log likelihood) of a vector of
/// data to a vector of existing scores.
pub trait AccumScore<X: Sync>: Rv<X> + Sync {
    // XXX: Default implementations can be improved upon by pre-computing
    // normalizers
    fn accum_score(&self, scores: &mut [f64], xs: &[X], present: &[bool]) {
        let xs_iter = xs.iter().zip(present.iter());
        scores.iter_mut().zip(xs_iter).for_each(|(score, (x, &r))| {
            // TODO: unnormed_loglike
            if r {
                *score += self.ln_f(x);
            }
        });
    }

    fn accum_score_par(&self, scores: &mut [f64], xs: &[X], present: &[bool]) {
        let xs_iter = xs.par_iter().zip_eq(present.par_iter());
        scores
            .par_iter_mut()
            .zip_eq(xs_iter)
            .for_each(|(score, (x, &r))| {
                // TODO: unnormed_loglike
                if r {
                    *score += self.ln_f(x);
                }
            });
    }

    // TODO: GPU implementation
}

// Since we don't care about the scores being normalized properly we can save
// some computation by not normalizing.
impl AccumScore<f64> for Gaussian {
    fn accum_score_par(
        &self,
        scores: &mut [f64],
        xs: &[f64],
        present: &[bool],
    ) {
        let mu = self.mu;
        let sigma = self.sigma;
        let log_z = -self.sigma.ln() - rv::consts::HALF_LN_2PI;

        let xs_iter = xs.par_iter().zip_eq(present.par_iter());
        scores
            .par_iter_mut()
            .zip_eq(xs_iter)
            .for_each(|(score, (x, &r))| {
                if r {
                    let term = (x - mu) / sigma;
                    let loglike = -0.5 * term * term + log_z;
                    *score += loglike;
                }
            });
    }
}

impl<X: CategoricalDatum> AccumScore<X> for Categorical {}
impl AccumScore<Label> for Labeler {}

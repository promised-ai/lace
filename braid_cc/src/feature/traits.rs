//! Defines the `Feature` trait for cross-categorization columns
use braid_data::label::Label;
use braid_data::FeatureData;
use braid_data::{Datum, SparseContainer};
use braid_stats::labeler::{Labeler, LabelerPrior};
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use braid_stats::MixtureType;
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};

use super::Component;
use crate::assignment::Assignment;
use crate::feature::{ColModel, Column, FType};

pub trait TranslateDatum<X>
where
    X: Clone + Default,
{
    /// Create an `X` from a `Datum`
    fn from_datum(datum: Datum) -> X;
    /// Convert an `X` into a `Datum`
    fn into_datum(x: X) -> Datum;

    /// Create a `SparseContainer` from a `FeatureData`
    fn from_feature_data(data: FeatureData) -> SparseContainer<X>;
    /// Convert a `SparseContainer` into a `FeatureData`
    fn into_feature_data(xs: SparseContainer<X>) -> FeatureData;

    /// Get the feature type
    fn ftype() -> FType;
}

/// A Cross-Categorization feature/column
#[enum_dispatch(ColModel)]
pub trait Feature {
    /// The feature ID
    fn id(&self) -> usize;
    /// Set the feature ID
    fn set_id(&mut self, id: usize);

    /// The number of rows
    fn len(&self) -> usize;
    /// Whether len is zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The number of components
    fn k(&self) -> usize;

    /// score each datum under component `k` and add to the corresponding
    /// entries in `scores`
    fn accum_score(&self, scores: &mut [f64], k: usize);
    /// Draw `k` components from the prior
    fn init_components(&mut self, k: usize, rng: &mut impl Rng);
    /// Redraw the component parameters from the posterior distribution,
    /// f(θ|x<sub>k</sub>).
    fn update_components(&mut self, rng: &mut impl Rng);
    /// Create new components and assign data to them according to the
    /// assignment.
    fn reassign(&mut self, asgn: &Assignment, rng: &mut impl Rng);
    /// The log likelihood of the datum in the Feature under the current
    /// assignment
    fn score(&self) -> f64;
    /// The log likelihood of the datum in the Feature under a different
    /// assignment
    fn asgn_score(&self, asgn: &Assignment) -> f64;
    /// Draw new prior parameters from the posterior, p(φ|θ). Returns the new
    /// log prior likelihood of the component parameters under the prior and
    /// the prior parameters under the hyperprior.
    fn update_prior_params(&mut self, rng: &mut impl Rng) -> f64;
    /// Draw an empty component from the prior and append it to the components
    /// vector
    fn append_empty_component(&mut self, rng: &mut impl Rng);
    /// Remove the component at index `k`
    fn drop_component(&mut self, k: usize);
    /// The log likelihood of the datum at `row_ix` under the component at
    /// index `k`
    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64>;
    /// The log posterior predictive function of the datum at `row_ix` under
    /// the component at index `k`
    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64;
    /// The marginal likelihood of the datum on its own
    fn singleton_score(&self, row_ix: usize) -> f64;
    /// The marginal likelihood of the data in component k
    fn logm(&self, k: usize) -> f64;

    /// Have the component at index `k` observe the datum at row `row_ix`
    fn observe_datum(&mut self, row_ix: usize, k: usize);
    /// Have the component at index `k` forget the datum at row `row_ix`
    fn forget_datum(&mut self, row_ix: usize, k: usize);

    /// Add an unassigned datum to the bottom of the feature
    fn append_datum(&mut self, x: Datum);
    /// Insert a Datum at a certain row index. If the `x` is `Missing`, removes
    /// the value and marks it as no present in the data container.
    fn insert_datum(&mut self, row_ix: usize, x: Datum);

    /// Get a datum
    fn datum(&self, ix: usize) -> Datum;
    /// Takes the data out of the column model as `FeatureData` and replaces it
    /// with an empty `SparseContainer`.
    fn take_data(&mut self) -> FeatureData;
    /// Take the datum at row_ix from component k and return the value if it is
    /// not `Missing`
    fn take_datum(&mut self, row_ix: usize, k: usize) -> Option<Datum>;
    /// Get a clone of the feature data
    fn clone_data(&self) -> FeatureData;
    /// Draw a sample from component `k`
    fn draw(&self, k: usize, rng: &mut impl Rng) -> Datum;
    /// Repopulate data on an empty feature
    fn repop_data(&mut self, data: FeatureData);
    /// Add the log probability of a datum to a weight vector.
    ///
    /// If `scaled` is `true`, the log probabilities will be scaled by the
    /// height of the mode, e.g., `logp(x) - logp(mode)`.
    fn accum_weights(
        &self,
        datum: &Datum,
        weights: &mut Vec<f64>,
        scaled: bool,
    );
    /// Get the Log PDF/PMF of `datum` under component `k`
    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64;
    fn ftype(&self) -> FType;

    /// Get a reference to the component at index k
    fn component(&self, k: usize) -> Component;

    /// Convert the component models into a mixture model
    fn to_mixture(&self, weights: Vec<f64>) -> MixtureType;

    /// Initialize the features from the prior and fill data for geweke
    fn geweke_init<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R);
}

#[enum_dispatch(ColModel)]
pub(crate) trait FeatureHelper: Feature {
    /// remove the datum at ix
    fn del_datum(&mut self, ix: usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment::AssignmentBuilder;
    use approx::*;
    use rv::dist::Gaussian;
    use rv::traits::Rv;

    #[test]
    fn score_and_asgn_score_equivalency() {
        let n_rows = 100;
        let mut rng = rand::thread_rng();
        let g = Gaussian::standard();
        let hyper = NixHyper::default();
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);
        for _ in 0..100 {
            let asgn = AssignmentBuilder::new(n_rows).build().unwrap();
            let xs: Vec<f64> = g.sample(n_rows, &mut rng);
            let data = SparseContainer::from(xs);
            let mut feature =
                Column::new(0, data, prior.clone(), hyper.clone());
            feature.reassign(&asgn, &mut rng);

            assert_relative_eq!(
                feature.score(),
                feature.asgn_score(&asgn),
                epsilon = 1E-8
            );
        }
    }
}

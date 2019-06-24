//! Defines the `Feature` trait for cross-categorization columns
use braid_stats::labeler::{Label, Labeler, LabelerPrior};
use braid_stats::prior::{Csd, Ng};
use braid_stats::MixtureType;
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rv::dist::{Categorical, Gaussian};

use super::{Component, FeatureData};
use crate::cc::assignment::Assignment;
use crate::cc::container::DataContainer;
use crate::cc::{ColModel, Column, Datum, FType};
use crate::result;

pub trait TranslateDatum<X>
where
    X: Clone,
{
    /// Create an `X` from a `Datum`
    fn from_datum(datum: Datum) -> X;
    /// Convert an `X` into a `Datum`
    fn into_datum(x: X) -> Datum;

    /// Create a `DataContainer` from a `FeatureData`
    fn from_feature_data(data: FeatureData) -> DataContainer<X>;
    /// Convert a `DataContainer` into a `FeatureData`
    fn into_feature_data(xs: DataContainer<X>) -> FeatureData;

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
    /// The number of components
    fn k(&self) -> usize;

    /// score each datum under component `k` and add to the corresponding
    /// entries in `scores`
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize);
    /// Draw `k` components from the prior
    fn init_components(&mut self, k: usize, rng: &mut impl Rng);
    /// Redraw the component parameters from the posterior distribution,
    /// f(θ|x<sub>k</sub>).
    fn update_components(&mut self, rng: &mut impl Rng);
    /// Create new components and assign data to them accoring to the
    /// assignment.
    fn reassign(&mut self, asgn: &Assignment, rng: &mut impl Rng);
    /// The log likelihood of the datum in the Feature under the current
    /// assignment
    fn score(&self) -> f64;
    /// The log likelihood of the datum in the Feature under a different
    /// assignment
    fn asgn_score(&self, asgn: &Assignment) -> f64;
    /// Draw new prior parameters from the posterior, p(φ|θ)
    fn update_prior_params(&mut self, rng: &mut impl Rng);
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

    /// Have the component at index `k` observe the datum at row `row_ix`
    fn observe_datum(&mut self, row_ix: usize, k: usize);
    /// Have the component at index `k` forget the datum at row `row_ix`
    fn forget_datum(&mut self, row_ix: usize, k: usize);

    /// Add an unassigned datum to the bottom of the feature
    fn append_datum(&mut self, x: Datum);

    /// Get a datum
    fn datum(&self, ix: usize) -> Datum;
    /// Takes the data out of the column model as `FeatureData` and replaces it
    /// with an empty `DataContainer`.
    fn take_data(&mut self) -> FeatureData;
    /// Get a clone of the feature data
    fn clone_data(&self) -> FeatureData;
    /// Draw a sample from component `k`
    fn draw(&self, k: usize, rng: &mut impl Rng) -> Datum;
    /// Repopulate data on an empty feature
    fn repop_data(&mut self, data: FeatureData) -> result::Result<()>;
    /// Add the log probability of a datum to a weight vector
    fn accum_weights(&self, datum: &Datum, weights: Vec<f64>) -> Vec<f64>;
    /// Get the Log PDF/PMF of `datum` under component `k`
    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64;
    fn ftype(&self) -> FType;

    /// Get a reference to the component at index k
    fn component(&self, k: usize) -> Component;

    /// Convert the component models into a mixture model
    fn to_mixture(&self) -> MixtureType;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::AssignmentBuilder;
    use approx::*;
    use braid_stats::prior::NigHyper;
    use rv::dist::Gaussian;
    use rv::traits::Rv;

    #[test]
    fn score_and_asgn_score_equivalency() {
        let nrows = 100;
        let mut rng = rand::thread_rng();
        let g = Gaussian::standard();
        let prior = Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::default());
        for _ in 0..100 {
            let asgn = AssignmentBuilder::new(nrows).build().unwrap();
            let xs: Vec<f64> = g.sample(nrows, &mut rng);
            let data = DataContainer::new(xs);
            let mut feature = Column::new(0, data, prior.clone());
            feature.reassign(&asgn, &mut rng);

            assert_relative_eq!(
                feature.score(),
                feature.asgn_score(&asgn),
                epsilon = 1E-8
            );
        }
    }
}

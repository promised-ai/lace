use super::{ColModel, Component, FType, Feature, FeatureHelper};
use crate::assignment::Assignment;
use lace_data::{Datum, FeatureData};
use lace_stats::MixtureType;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Latent {
    pub column: Box<ColModel>,
    pub assignment: Vec<usize>,
}

impl FeatureHelper for Latent {
    fn del_datum(&mut self, ix: usize) {
        self.column.del_datum(ix);
    }
}

impl Feature for Latent {
    #[inline]
    fn id(&self) -> usize {
        self.column.id()
    }

    #[inline]
    fn set_id(&mut self, id: usize) {
        self.column.set_id(id)
    }

    #[inline]
    fn accum_score(&self, scores: &mut [f64], k: usize) {
        self.column.accum_score(scores, k);
    }

    #[inline]
    fn len(&self) -> usize {
        self.column.len()
    }

    #[inline]
    fn k(&self) -> usize {
        self.column.k()
    }

    #[inline]
    fn init_components(&mut self, k: usize, rng: &mut impl Rng) {
        self.column.init_components(k, rng);
    }

    #[inline]
    fn update_components(&mut self, rng: &mut impl Rng) {
        self.column.update_components(rng);
    }

    #[inline]
    fn reassign(&mut self, asgn: &Assignment, rng: &mut impl Rng) {
        self.column.reassign(asgn, rng);
        self.assignment = asgn.asgn.clone();
    }

    #[inline]
    fn score(&self) -> f64 {
        self.column.score()
    }

    #[inline]
    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        self.column.asgn_score(asgn)
    }

    #[inline]
    fn update_prior_params(&mut self, rng: &mut impl Rng) -> f64 {
        self.column.update_prior_params(rng)
    }

    #[inline]
    fn append_empty_component(&mut self, rng: &mut impl Rng) {
        self.column.append_empty_component(rng);
    }

    fn drop_component(&mut self, k: usize) {
        self.column.drop_component(k);
    }

    #[inline]
    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        self.column.predictive_score_at(row_ix, k)
    }

    #[inline]
    fn logm(&self, k: usize) -> f64 {
        self.column.logm(k)
    }

    #[inline]
    fn singleton_score(&self, row_ix: usize) -> f64 {
        self.column.singleton_score(row_ix)
    }

    #[inline]
    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        self.assignment[row_ix] = k;
        self.column.observe_datum(row_ix, k);
    }

    #[inline]
    fn take_datum(&mut self, row_ix: usize, k: usize) -> Option<Datum> {
        // Latent columns should never return data
        let _ = self.column.take_datum(row_ix, k);
        None
    }

    #[inline]
    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        self.assignment[row_ix] = std::usize::MAX; // unassigned
        self.column.forget_datum(row_ix, k);
    }

    #[inline]
    fn append_datum(&mut self, x: Datum) {
        self.column.append_datum(Datum::Binary(!x.is_missing()));
    }

    #[inline]
    fn insert_datum(&mut self, row_ix: usize, x: Datum) {
        self.column.insert_datum(row_ix, x);
    }

    #[inline]
    fn is_missing(&self, ix: usize) -> bool {
        let is_missing = self.column.is_missing(ix);
        assert!(!is_missing);
        is_missing
    }

    #[inline]
    fn datum(&self, ix: usize) -> Datum {
        self.column.datum(ix)
    }

    fn take_data(&mut self) -> FeatureData {
        // just drop the data
        let _ = self.column.take_data();
        FeatureData::Latent {
            seed: 1337,
            len: self.len(),
        }
    }

    fn clone_data(&self) -> FeatureData {
        self.column.clone_data()
    }

    fn draw(&self, k: usize, rng: &mut impl Rng) -> Datum {
        self.column.draw(k, rng)
    }

    fn repop_data(&mut self, data: FeatureData) {
        use rand::SeedableRng;
        // repopulate by drawing
        let mut rng = if let FeatureData::Latent { seed, .. } = data {
            rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed)
        } else {
            panic!("Latent column '{}' expected FeatureData::Latent", self.id())
        };

        for (row_ix, &k) in self.assignment.iter().enumerate() {
            let x = self.draw(k, &mut rng);
            self.column.insert_datum(row_ix, x)
        }
    }

    fn accum_weights(
        &self,
        datum: &Datum,
        weights: &mut Vec<f64>,
        scaled: bool,
    ) {
        self.column.accum_weights(datum, weights, scaled)
    }

    fn accum_exp_weights(&self, datum: &Datum, weights: &mut Vec<f64>) {
        self.column.accum_exp_weights(datum, weights)
    }

    #[inline]
    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64 {
        self.column.cpnt_logp(datum, k)
    }

    #[inline]
    fn cpnt_likelihood(&self, datum: &Datum, k: usize) -> f64 {
        self.column.cpnt_likelihood(datum, k)
    }

    #[inline]
    fn ftype(&self) -> FType {
        self.column.ftype()
    }

    #[inline]
    fn is_latent(&self) -> bool {
        true
    }

    #[inline]
    fn component(&self, k: usize) -> Component {
        self.column.component(k)
    }

    fn to_mixture(&self, weights: Vec<f64>) -> MixtureType {
        self.column.to_mixture(weights)
    }

    fn geweke_init<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) {
        self.column.geweke_init(asgn, rng);
    }
}

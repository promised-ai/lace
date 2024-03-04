use super::{ColModel, Column, Component, FType, Feature, FeatureHelper};
use lace_data::{Datum, FeatureData, SparseContainer};
use lace_stats::assignment::Assignment;
use lace_stats::rv::dist::{Bernoulli, Beta};
use lace_stats::MixtureType;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Missing-not-at-random column type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingNotAtRandom {
    /// The data model for this column
    pub fx: Box<ColModel>,
    /// The model of data presence (true if present)
    pub present: Column<bool, Bernoulli, Beta, ()>,
}

impl FeatureHelper for MissingNotAtRandom {
    fn del_datum(&mut self, ix: usize) {
        self.fx.del_datum(ix);
        self.present.del_datum(ix);
    }
}

impl Feature for MissingNotAtRandom {
    #[inline]
    fn id(&self) -> usize {
        self.fx.id()
    }

    #[inline]
    fn set_id(&mut self, id: usize) {
        self.fx.set_id(id)
    }

    #[inline]
    fn accum_score(&self, scores: &mut [f64], k: usize) {
        self.fx.accum_score(scores, k);
        self.present.accum_score(scores, k);
    }

    #[inline]
    fn len(&self) -> usize {
        self.fx.len()
    }

    #[inline]
    fn k(&self) -> usize {
        self.fx.k()
    }

    #[inline]
    fn init_components(&mut self, k: usize, rng: &mut impl Rng) {
        self.fx.init_components(k, rng);
        self.present.init_components(k, rng);
    }

    #[inline]
    fn update_components(&mut self, rng: &mut impl Rng) {
        self.fx.update_components(rng);
        self.present.update_components(rng);
    }

    #[inline]
    fn reassign(&mut self, asgn: &Assignment, rng: &mut impl Rng) {
        self.fx.reassign(asgn, rng);
        self.present.reassign(asgn, rng);
    }

    #[inline]
    fn score(&self) -> f64 {
        self.fx.score() + self.present.score()
    }

    #[inline]
    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        self.fx.asgn_score(asgn) + self.present.asgn_score(asgn)
    }

    #[inline]
    fn update_prior_params(&mut self, rng: &mut impl Rng) -> f64 {
        self.fx.update_prior_params(rng)
    }

    #[inline]
    fn append_empty_component(&mut self, rng: &mut impl Rng) {
        self.fx.append_empty_component(rng);
        self.present.append_empty_component(rng);
    }

    fn drop_component(&mut self, k: usize) {
        self.fx.drop_component(k);
        self.present.drop_component(k);
    }

    #[inline]
    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        self.fx.predictive_score_at(row_ix, k)
            + self.present.predictive_score_at(row_ix, k)
    }

    #[inline]
    fn logm(&self, k: usize) -> f64 {
        self.fx.logm(k) + self.present.logm(k)
    }

    #[inline]
    fn singleton_score(&self, row_ix: usize) -> f64 {
        self.fx.singleton_score(row_ix) + self.present.singleton_score(row_ix)
    }

    #[inline]
    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        self.fx.observe_datum(row_ix, k);
        self.present.observe_datum(row_ix, k);
    }

    #[inline]
    fn take_datum(&mut self, row_ix: usize, k: usize) -> Option<Datum> {
        self.fx.take_datum(row_ix, k).map(|x| {
            // if the datum was present, we must now mark it as missing
            self.present.insert_datum(row_ix, Datum::Binary(false));
            x
        })
    }

    #[inline]
    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        self.fx.forget_datum(row_ix, k);
        self.present.forget_datum(row_ix, k);
    }

    #[inline]
    fn append_datum(&mut self, x: Datum) {
        self.present.append_datum(Datum::Binary(!x.is_missing()));
        self.fx.append_datum(x);
    }

    #[inline]
    fn insert_datum(&mut self, row_ix: usize, x: Datum) {
        self.present
            .insert_datum(row_ix, Datum::Binary(!x.is_missing()));
        self.fx.insert_datum(row_ix, x);
    }

    #[inline]
    fn is_missing(&self, ix: usize) -> bool {
        self.fx.is_missing(ix)
    }

    #[inline]
    fn datum(&self, ix: usize) -> Datum {
        self.fx.datum(ix)
    }

    fn take_data(&mut self) -> FeatureData {
        let _ = self.present.take_data();
        self.fx.take_data()
    }

    fn clone_data(&self) -> FeatureData {
        self.fx.clone_data()
    }

    fn draw(&self, k: usize, rng: &mut impl Rng) -> Datum {
        if let Datum::Binary(true) = self.present.draw(k, rng) {
            self.fx.draw(k, rng)
        } else {
            Datum::Missing
        }
    }

    fn repop_data(&mut self, data: FeatureData) {
        let n = data.len();
        let present =
            (0..n).map(|ix| data.is_present(ix)).collect::<Vec<bool>>();
        self.present.data = SparseContainer::from(present);
        self.fx.repop_data(data);
    }

    fn accum_weights(
        &self,
        datum: &Datum,
        weights: &mut Vec<f64>,
        scaled: bool,
    ) {
        self.present.accum_weights(
            &Datum::Binary(!datum.is_missing()),
            weights,
            scaled,
        );
        if !datum.is_missing() {
            self.fx.accum_weights(datum, weights, scaled);
        }
    }

    fn accum_exp_weights(&self, datum: &Datum, weights: &mut Vec<f64>) {
        self.present
            .accum_exp_weights(&Datum::Binary(!datum.is_missing()), weights);
        if !datum.is_missing() {
            self.fx.accum_exp_weights(datum, weights);
        }
    }

    #[inline]
    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64 {
        if datum.is_missing() {
            self.present.cpnt_logp(&Datum::Binary(false), k)
        } else {
            self.fx.cpnt_logp(datum, k)
        }
    }

    #[inline]
    fn cpnt_likelihood(&self, datum: &Datum, k: usize) -> f64 {
        if datum.is_missing() {
            self.present.cpnt_likelihood(&Datum::Binary(false), k)
        } else {
            self.fx.cpnt_likelihood(datum, k)
        }
    }

    #[inline]
    fn ftype(&self) -> FType {
        self.fx.ftype()
    }

    #[inline]
    fn component(&self, k: usize) -> Component {
        // FIXME: is this right?
        self.fx.component(k)
    }

    fn to_mixture(&self, weights: Vec<f64>) -> MixtureType {
        // FIXME: is this right?
        self.fx.to_mixture(weights)
    }

    fn geweke_init<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) {
        self.fx.geweke_init(asgn, rng);
        self.present.geweke_init(asgn, rng);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use lace_data::{Category, Container};

    // Return categorical (k = 4) column with missing values at indices 50, 51,
    // and 52.
    fn mnar_col() -> (MissingNotAtRandom, Assignment) {
        let n: usize = 100;
        let mut data = SparseContainer::from(
            (0..4_u8).cycle().take(n).collect::<Vec<u8>>(),
        );
        let _ = data.set_missing(50);
        let _ = data.set_missing(51);
        let _ = data.set_missing(52);

        let fx = {
            let hyper = lace_stats::prior::csd::CsdHyper::new(1.0, 1.0);
            let prior = lace_stats::prior::csd::vague(4);
            let column = Column::new(0, data, prior, hyper);
            ColModel::Categorical(column)
        };

        let present = {
            let data_p: Vec<bool> =
                (0..n).map(|ix| fx.is_present(ix)).collect();
            let data_p = SparseContainer::from(data_p);
            Column::new(fx.id(), data_p, Beta::jeffreys(), ())
        };

        let mut col = MissingNotAtRandom {
            fx: Box::new(fx),
            present,
        };
        let mut rng = rand::thread_rng();
        let asgn = lace_stats::prior_process::Builder::new(n)
            .seed_from_rng(&mut rng)
            .build()
            .unwrap()
            .asgn;
        col.reassign(&asgn, &mut rng);
        (col, asgn)
    }

    #[test]
    fn cpnt_logp_present() {
        let (col, _) = mnar_col();

        let f0 = col.cpnt_logp(&Datum::Categorical(Category::U8(0)), 0);
        let f1 = col.cpnt_logp(&Datum::Categorical(Category::U8(1)), 0);
        let f2 = col.cpnt_logp(&Datum::Categorical(Category::U8(2)), 0);
        let f3 = col.cpnt_logp(&Datum::Categorical(Category::U8(3)), 0);

        println!("{:?}", [f0, f1, f2, f3]);

        assert_relative_eq!(
            lace_consts::rv::misc::logsumexp(&[f0, f1, f2, f3]).exp(),
            1.0,
            epsilon = 1e-10
        )
    }

    #[test]
    fn cpnt_logp_missing() {
        let (col, _) = mnar_col();

        let f_missing = col.cpnt_logp(&Datum::Missing, 0);

        assert!(f_missing < 0.2_f64.ln());
    }

    #[test]
    fn cpnt_likelihood_present() {
        let (col, _) = mnar_col();

        let f0 = col.cpnt_likelihood(&Datum::Categorical(0_u8.into()), 0);
        let f1 = col.cpnt_likelihood(&Datum::Categorical(1_u8.into()), 0);
        let f2 = col.cpnt_likelihood(&Datum::Categorical(2_u8.into()), 0);
        let f3 = col.cpnt_likelihood(&Datum::Categorical(3_u8.into()), 0);

        assert_relative_eq!(f0 + f1 + f2 + f3, 1.0, epsilon = 1e-10)
    }

    #[test]
    fn cpnt_likelihood_missing() {
        let (col, _) = mnar_col();

        let f_missing = col.cpnt_likelihood(&Datum::Missing, 0);

        assert!(f_missing < 0.2);
    }

    #[test]
    fn remove_data_present() {
        let (mut col, asgn) = mnar_col();

        let ix = 1;
        assert_eq!(col.present.data.get(ix), Some(true));

        let k = asgn.asgn[ix];
        let x = col.take_datum(ix, k);

        assert!(x.is_some());
        assert_eq!(col.present.data.get(ix), Some(false));
        assert_eq!(col.take_datum(ix, k), None);
    }

    #[test]
    fn remove_data_missing() {
        let (mut col, asgn) = mnar_col();

        let ix = 51;
        assert_eq!(col.present.data.get(ix), Some(false));

        let k = asgn.asgn[ix];
        let x = col.take_datum(ix, k);

        assert!(x.is_none());
        assert_eq!(col.present.data.get(ix), Some(false));
    }

    #[test]
    fn insert_data_present_into_present() {
        let (mut col, _) = mnar_col();

        let ix = 1;

        assert_eq!(col.fx.datum(ix), Datum::Categorical(1_u8.into()));
        assert_eq!(col.present.data.get(ix), Some(true));

        col.insert_datum(ix, Datum::Categorical(0_u8.into()));

        assert_eq!(col.fx.datum(ix), Datum::Categorical(0_u8.into()));
        assert_eq!(col.present.data.get(ix), Some(true));
    }

    #[test]
    fn insert_data_missing_into_present() {
        let (mut col, _) = mnar_col();

        let ix = 1;

        assert_eq!(col.fx.datum(ix), Datum::Categorical(1_u8.into()));
        assert_eq!(col.present.data.get(ix), Some(true));

        col.insert_datum(ix, Datum::Missing);

        assert_eq!(col.fx.datum(ix), Datum::Missing);
        assert_eq!(col.present.data.get(ix), Some(false));
    }

    #[test]
    fn insert_data_present_into_missing() {
        let (mut col, _) = mnar_col();

        let ix = 51;

        assert_eq!(col.fx.datum(ix), Datum::Missing);
        assert_eq!(col.present.data.get(ix), Some(false));

        col.insert_datum(ix, Datum::Categorical(0_u8.into()));

        assert_eq!(col.fx.datum(ix), Datum::Categorical(0_u8.into()));
        assert_eq!(col.present.data.get(ix), Some(true));
    }

    #[test]
    fn insert_data_missing_into_missing() {
        let (mut col, _) = mnar_col();

        let ix = 51;

        assert_eq!(col.fx.datum(ix), Datum::Missing);
        assert_eq!(col.present.data.get(ix), Some(false));

        col.insert_datum(ix, Datum::Missing);

        assert_eq!(col.fx.datum(ix), Datum::Missing);
        assert_eq!(col.present.data.get(ix), Some(false));
    }
}

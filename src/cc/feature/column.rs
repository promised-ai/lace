use std::mem;

use braid_stats::labeler::{Label, Labeler, LabelerPrior};
use braid_stats::prior::{Csd, Ng};
use braid_utils::misc::minmax;
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::dist::{Categorical, Gaussian};
use rv::traits::{Rv, SuffStat};
use serde::{Deserialize, Serialize};

use super::FeatureData;
use crate::cc::container::DataContainer;
use crate::cc::feature::traits::{Feature, TranslateDatum};
use crate::cc::{Assignment, ConjugateComponent, Datum, FType};
use crate::dist::traits::AccumScore;
use crate::dist::{BraidDatum, BraidLikelihood, BraidPrior, BraidStat};
use crate::result;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
/// A partitioned columns of data
pub struct Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    pub id: usize,
    // TODO: Figure out a way to optionally serialize data
    pub data: DataContainer<X>,
    pub components: Vec<ConjugateComponent<X, Fx>>,
    pub prior: Pr,
    // TODO: pointers to data on GPU
}

#[enum_dispatch]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ColModel {
    Continuous(Column<f64, Gaussian, Ng>),
    Categorical(Column<u8, Categorical, Csd>),
    Labeler(Column<Label, Labeler, LabelerPrior>),
    // Binary(Column<bool, Bernoulli, BetaBernoulli),
}

impl ColModel {
    pub fn impute_bounds(&self) -> Option<(f64, f64)> {
        match self {
            ColModel::Continuous(ftr) => {
                let means: Vec<f64> =
                    ftr.components.iter().map(|cpnt| cpnt.fx.mu).collect();
                Some(minmax(&means))
            }
            _ => None,
        }
    }
}

impl<X, Fx, Pr> Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    pub fn new(id: usize, data: DataContainer<X>, prior: Pr) -> Self {
        Column {
            id,
            data,
            components: Vec::new(),
            prior,
        }
    }

    pub fn len(&self) -> usize {
        // XXX: this will fail on features with dropped data
        self.data.len()
    }

    pub fn components(&self) -> Vec<Fx> {
        self.components.iter().map(|cpnt| cpnt.fx.clone()).collect()
    }
}

macro_rules! impl_translate_datum {
    ($x:ty, $fx:ty, $pr:ty, $datum_variant:ident) => {
        impl_translate_datum!($x, $fx, $pr, $datum_variant, $datum_variant);
    };
    ($x:ty, $fx:ty, $pr:ty, $datum_variant:ident, $fdata_variant:ident) => {
        impl TranslateDatum<$x> for Column<$x, $fx, $pr> {
            fn from_datum(datum: Datum) -> $x {
                match datum {
                    Datum::$datum_variant(x) => x,
                    _ => panic!("Invalid Datum variant for conversion"),
                }
            }

            fn into_datum(x: $x) -> Datum {
                Datum::$datum_variant(x)
            }

            fn from_feature_data(data: FeatureData) -> DataContainer<$x> {
                match data {
                    FeatureData::$fdata_variant(xs) => xs,
                    _ => panic!("Invalid FeatureData variant for conversion"),
                }
            }

            fn into_feature_data(xs: DataContainer<$x>) -> FeatureData {
                FeatureData::$fdata_variant(xs)
            }

            fn ftype() -> FType {
                FType::$fdata_variant
            }
        }
    };
}

impl_translate_datum!(f64, Gaussian, Ng, Continuous);
impl_translate_datum!(u8, Categorical, Csd, Categorical);
impl_translate_datum!(Label, Labeler, LabelerPrior, Label, Labeler);

fn draw_cpnts<X, Fx, Pr>(
    prior: &Pr,
    k: usize,
    mut rng: &mut impl Rng,
) -> Vec<ConjugateComponent<X, Fx>>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    (0..k)
        .map(|_| ConjugateComponent::new(prior.draw(&mut rng)))
        .collect()
}

#[allow(dead_code)]
impl<X, Fx, Pr> Feature for Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
    Self: TranslateDatum<X>,
{
    fn id(&self) -> usize {
        self.id
    }

    fn set_id(&mut self, id: usize) {
        self.id = id
    }

    fn accum_score(&self, mut scores: &mut Vec<f64>, k: usize) {
        // TODO: Decide when to use parallel or GPU
        self.components[k].accum_score(
            &mut scores,
            &self.data.data,
            &self.data.present,
        );
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn k(&self) -> usize {
        self.components.len()
    }

    fn init_components(&mut self, k: usize, mut rng: &mut impl Rng) {
        self.components = draw_cpnts(&self.prior, k, &mut rng);
    }

    fn update_components(&mut self, mut rng: &mut impl Rng) {
        let prior = self.prior.clone();
        self.components.iter_mut().for_each(|cpnt| {
            cpnt.fx = prior.posterior(&cpnt.obs()).draw(&mut rng);
        })
    }

    fn reassign(&mut self, asgn: &Assignment, mut rng: &mut impl Rng) {
        // re-draw empty k componants.
        // TODO: We should consider a way to do this without drawing from the
        // prior because we're just going to overwrite what we draw in a fe
        // lines. Wasted cycles.
        let mut components = draw_cpnts(&self.prior, asgn.ncats, &mut rng);

        // have the component obseve all their data
        self.data
            .zip()
            .zip(asgn.iter())
            .for_each(|((x, present), z)| {
                if *present {
                    components[*z].observe(x)
                }
            });

        // Set the components
        self.components = components;

        // update the component according to the posterior
        self.update_components(&mut rng);
    }

    fn score(&self) -> f64 {
        self.components
            .iter()
            .fold(0.0, |acc, cpnt| acc + self.prior.ln_m(&cpnt.obs()))
    }

    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        let xks = self.data.group_by(asgn);
        xks.iter().fold(0.0, |acc, xk| {
            let data = DataOrSuffStat::Data(xk);
            acc + self.prior.ln_m(&data)
        })
    }

    fn update_prior_params(&mut self, mut rng: &mut impl Rng) {
        let components: Vec<&Fx> =
            self.components.iter().map(|cpnt| &cpnt.fx).collect();
        self.prior.update_prior(&components, &mut rng);
    }

    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
        let cpnt = ConjugateComponent::new(self.prior.draw(&mut rng));
        self.components.push(cpnt);
    }

    fn drop_component(&mut self, k: usize) {
        // cpnt goes out of scope and is dropped (Hopefully)
        let _cpnt = self.components.remove(k);
    }

    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64> {
        if self.data.present[row_ix] {
            let x = &self.data.data[row_ix];
            Some(self.components[k].ln_f(x))
        } else {
            None
        }
    }

    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        if self.data.present[row_ix] {
            self.prior
                .ln_pp(&self.data[row_ix], &self.components[k].obs())
        } else {
            0.0
        }
    }

    fn singleton_score(&self, row_ix: usize) -> f64 {
        if self.data.present[row_ix] {
            let mut stat = self.components[0].fx.empty_suffstat();
            stat.observe(&self.data.data[row_ix]);
            self.prior.ln_m(&DataOrSuffStat::SuffStat(&stat))
        } else {
            0.0
        }
    }

    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        if self.data.present[row_ix] {
            let x = &self.data[row_ix];
            self.components[k].observe(x);
        }
    }

    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        if self.data.present[row_ix] {
            let x = &self.data[row_ix];
            self.components[k].forget(x);
        }
    }

    fn append_datum(&mut self, x: Datum) {
        self.data.push_datum(x);
    }

    fn datum(&self, ix: usize) -> Datum {
        if self.data.present[ix] {
            Self::into_datum(self.data.data[ix].clone())
        } else {
            Datum::Missing
        }
    }

    fn take_data(&mut self) -> FeatureData {
        let mut data: DataContainer<X> = DataContainer::empty();
        mem::swap(&mut data, &mut self.data);
        Self::into_feature_data(data)
    }

    fn clone_data(&self) -> FeatureData {
        Self::into_feature_data(self.data.clone())
    }

    fn draw(&self, k: usize, mut rng: &mut impl Rng) -> Datum {
        let x: X = self.components[k].draw(&mut rng);
        Self::into_datum(x)
    }

    fn repop_data(&mut self, data: FeatureData) -> result::Result<()> {
        let mut xs = Self::from_feature_data(data);
        mem::swap(&mut xs, &mut self.data);
        Ok(())
    }

    fn accum_weights(&self, datum: &Datum, mut weights: Vec<f64>) -> Vec<f64> {
        if self.components.len() != weights.len() {
            let msg = format!(
                "Weights: {:?}, n_components: {}",
                weights,
                self.components.len()
            );
            panic!(msg);
        }

        let x: X = Self::from_datum(datum.clone());

        self.components
            .iter()
            .enumerate()
            .for_each(|(k, c)| weights[k] += c.ln_f(&x));

        weights
    }

    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64 {
        let x: X = Self::from_datum(datum.to_owned());
        self.components[k].ln_f(&x)
    }

    fn ftype(&self) -> FType {
        <Self as TranslateDatum<X>>::ftype()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cc::{
        AssignmentBuilder, Column, DataContainer, Feature, FeatureData,
    };
    use braid_stats::prior::NigHyper;

    fn gauss_fixture() -> ColModel {
        let mut rng = rand::thread_rng();
        let asgn = AssignmentBuilder::new(5)
            .with_alpha(1.0)
            .flat()
            .build()
            .unwrap();
        let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let hyper = NigHyper::default();
        let data = DataContainer::new(data_vec);
        let prior = Ng::new(0.0, 1.0, 1.0, 1.0, hyper);

        let mut col = Column::new(0, data, prior);
        col.reassign(&asgn, &mut rng);
        ColModel::Continuous(col)
    }

    fn categorical_fixture_u8() -> ColModel {
        let mut rng = rand::thread_rng();
        let asgn = AssignmentBuilder::new(5)
            .with_alpha(1.0)
            .flat()
            .build()
            .unwrap();
        let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
        let data = DataContainer::new(data_vec);
        let prior = Csd::vague(3, &mut rng);

        let mut col = Column::new(0, data, prior);
        col.reassign(&asgn, &mut rng);
        ColModel::Categorical(col)
    }

    #[test]
    fn take_continuous_data_should_leave_col_model_data_empty() {
        let mut col_model = gauss_fixture();
        let data = col_model.take_data();
        match data {
            FeatureData::Continuous(d) => assert_eq!(d.len(), 5),
            _ => panic!("Returned wrong FeatureData type."),
        }
        match col_model {
            ColModel::Continuous(f) => assert_eq!(f.data.len(), 0),
            _ => panic!("Returned wrong ColModel type."),
        }
    }

    #[test]
    fn take_categorical_data_should_leave_col_model_data_empty() {
        let mut col_model = categorical_fixture_u8();
        let data = col_model.take_data();
        match data {
            FeatureData::Categorical(d) => assert_eq!(d.len(), 5),
            _ => panic!("Returned wrong FeatureData type."),
        }
        match col_model {
            ColModel::Categorical(f) => assert_eq!(f.data.len(), 0),
            _ => panic!("Returned wrong ColModel type."),
        }
    }

    #[test]
    fn repop_categorical_data_should_put_the_data_back_in() {
        let mut col_model = categorical_fixture_u8();
        let data = col_model.take_data();
        match col_model {
            ColModel::Categorical(ref f) => assert_eq!(f.data.len(), 0),
            _ => panic!("Returned wrong ColModel type."),
        };
        col_model.repop_data(data).expect("Could not repop");
        match col_model {
            ColModel::Categorical(ref f) => assert_eq!(f.data.len(), 5),
            _ => panic!("Returned wrong ColModel type."),
        };
    }

    #[test]
    fn repop_continuous_data_should_put_the_data_back_in() {
        let mut col_model = gauss_fixture();
        let data = col_model.take_data();
        match col_model {
            ColModel::Continuous(ref f) => assert_eq!(f.data.len(), 0),
            _ => panic!("Returned wrong ColModel type."),
        };
        col_model.repop_data(data).expect("Could not repop");
        match col_model {
            ColModel::Continuous(ref f) => assert_eq!(f.data.len(), 5),
            _ => panic!("Returned wrong ColModel type."),
        };
    }
}

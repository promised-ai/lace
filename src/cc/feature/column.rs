use std::mem;
use std::vec::Drain;

use braid_stats::labeler::{Label, Labeler, LabelerPrior};
use braid_stats::prior::{Csd, Ng, Pg};
use braid_stats::MixtureType;
use braid_stats::QmcEntropy;
use braid_utils::MinMax;
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rv::data::DataOrSuffStat;
use rv::dist::{Categorical, Gaussian, Mixture, Poisson};
use rv::traits::{Mean, QuadBounds, Rv, SuffStat};
use serde::{Deserialize, Serialize};

use super::{Component, FeatureData};
use crate::cc::container::DataContainer;
use crate::cc::feature::traits::{Feature, TranslateDatum};
use crate::cc::{Assignment, ConjugateComponent, Datum, FType};
use crate::dist::traits::AccumScore;
use crate::dist::{BraidDatum, BraidLikelihood, BraidPrior, BraidStat};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
/// A partitioned columns of data
pub struct Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    MixtureType: From<Mixture<Fx>>,
    Fx::Stat: BraidStat,
{
    pub id: usize,
    pub data: DataContainer<X>,
    pub components: Vec<ConjugateComponent<X, Fx>>,
    pub prior: Pr,
}

#[enum_dispatch]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ColModel {
    Continuous(Column<f64, Gaussian, Ng>),
    Categorical(Column<u8, Categorical, Csd>),
    Labeler(Column<Label, Labeler, LabelerPrior>),
    Count(Column<u32, Poisson, Pg>),
}

impl ColModel {
    /// Get impute bounds for univariate continuous distributions
    pub fn impute_bounds(&self) -> Option<(f64, f64)> {
        match self {
            ColModel::Continuous(ftr) => {
                ftr.components.iter().map(|cpnt| cpnt.fx.mu()).minmax()
            }
            ColModel::Count(ftr) => ftr
                .components
                .iter()
                .map(|cpnt| {
                    let mean: f64 =
                        cpnt.fx.mean().expect("Poisson always has a mean");
                    mean
                })
                .minmax()
                .map(|(lower, upper)| {
                    ((lower.floor() - 1.0).max(0.0), upper.ceil())
                }),
            _ => None,
        }
    }
}

impl<X, Fx, Pr> Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    MixtureType: From<Mixture<Fx>>,
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

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn components(&self) -> &Vec<ConjugateComponent<X, Fx>> {
        &self.components
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
impl_translate_datum!(u32, Poisson, Pg, Count);
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
    MixtureType: From<Mixture<Fx>>,
    Self: TranslateDatum<X>,
{
    #[inline]
    fn id(&self) -> usize {
        self.id
    }

    #[inline]
    fn set_id(&mut self, id: usize) {
        self.id = id
    }

    #[inline]
    fn accum_score(&self, mut scores: &mut [f64], k: usize) {
        // TODO: Decide when to use parallel or GPU
        self.components[k].accum_score_par(
            &mut scores,
            &self.data.data,
            &self.data.present,
        );
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn k(&self) -> usize {
        self.components.len()
    }

    #[inline]
    fn init_components(&mut self, k: usize, mut rng: &mut impl Rng) {
        self.components = draw_cpnts(&self.prior, k, &mut rng);
    }

    #[inline]
    fn update_components(&mut self, mut rng: &mut impl Rng) {
        let prior = self.prior.clone();
        self.components.iter_mut().for_each(|cpnt| {
            cpnt.fx = prior.posterior(&cpnt.obs()).draw(&mut rng);
        })
    }

    #[inline]
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

    #[inline]
    fn score(&self) -> f64 {
        self.components
            .iter()
            .fold(0.0, |acc, cpnt| acc + self.prior.ln_m(&cpnt.obs()))
    }

    #[inline]
    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        let empty_stat = self.prior.empty_suffstat();

        let mut stats: Vec<_> =
            (0..asgn.ncats).map(|_| empty_stat.clone()).collect();

        asgn.iter()
            .zip(self.data.zip())
            .for_each(|(&z, (x, &present))| {
                if present {
                    stats[z].observe(x)
                }
            });

        stats.iter().fold(0_f64, |acc, stat| {
            let data = DataOrSuffStat::SuffStat(stat);
            acc + self.prior.ln_m(&data)
        })
    }

    #[inline]
    fn update_prior_params(&mut self, mut rng: &mut impl Rng) -> f64 {
        let components: Vec<&Fx> =
            self.components.iter().map(|cpnt| &cpnt.fx).collect();
        self.prior.update_prior(&components, &mut rng)
    }

    #[inline]
    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
        let cpnt = ConjugateComponent::new(self.prior.draw(&mut rng));
        self.components.push(cpnt);
    }

    fn drop_component(&mut self, k: usize) {
        // cpnt goes out of scope and is dropped
        let _cpnt = self.components.remove(k);
    }

    #[inline]
    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64> {
        if self.data.present[row_ix] {
            let x = &self.data.data[row_ix];
            Some(self.components[k].ln_f(x))
        } else {
            None
        }
    }

    #[inline]
    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        if self.data.present[row_ix] {
            self.prior
                .ln_pp(&self.data[row_ix], &self.components[k].obs())
        } else {
            0.0
        }
    }

    #[inline]
    fn logm(&self, k: usize) -> f64 {
        self.prior.ln_m(&self.components[k].obs())
    }

    #[inline]
    fn singleton_score(&self, row_ix: usize) -> f64 {
        if self.data.present[row_ix] {
            let mut stat = self.prior.empty_suffstat();
            stat.observe(&self.data.data[row_ix]);
            self.prior.ln_m(&DataOrSuffStat::SuffStat(&stat))
        } else {
            0.0
        }
    }

    #[inline]
    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        if self.data.present[row_ix] {
            let x = &self.data[row_ix];
            self.components[k].observe(x);
        }
    }

    #[inline]
    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        if self.data.present[row_ix] {
            let x = &self.data[row_ix];
            self.components[k].forget(x);
        }
    }

    #[inline]
    fn append_datum(&mut self, x: Datum) {
        self.data.push_datum(x);
    }

    #[inline]
    fn insert_datum(&mut self, row_ix: usize, x: Datum) {
        self.data.insert_datum(row_ix, x);
    }

    #[inline]
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

    fn repop_data(&mut self, data: FeatureData) {
        let mut xs = Self::from_feature_data(data);
        mem::swap(&mut xs, &mut self.data);
    }

    fn accum_weights(
        &self,
        datum: &Datum,
        weights: &mut Vec<f64>,
        scaled: bool,
    ) {
        if self.components.len() != weights.len() {
            let msg = format!(
                "Weights: {:?}, n_components: {}",
                weights,
                self.components.len()
            );
            panic!(msg);
        }

        let x: X = Self::from_datum(datum.clone());

        weights
            .iter_mut()
            .zip(self.components.iter())
            .for_each(|(w, c)| {
                let ln_fx = c.ln_f(&x);
                if scaled {
                    // Scale by the height of the mode. The component mode must
                    // always be defined.
                    let mode: X = c.fx.mode().unwrap();
                    let ln_fmode = c.ln_f(&mode);

                    *w += ln_fx - ln_fmode;
                } else {
                    *w += ln_fx;
                }
            });
    }

    #[inline]
    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64 {
        let x: X = Self::from_datum(datum.to_owned());
        self.components[k].ln_f(&x)
    }

    #[inline]
    fn ftype(&self) -> FType {
        <Self as TranslateDatum<X>>::ftype()
    }

    #[inline]
    fn component(&self, k: usize) -> Component {
        // TODO: would be nive to return a reference
        self.components[k].clone().into()
    }

    fn to_mixture(&self, mut weights: Vec<f64>) -> MixtureType {
        // Do not include components with zero-valued weights
        let components: Vec<Fx> = self
            .components
            .iter()
            .zip(weights.iter())
            .filter_map(|(cpnt, &weight)| {
                if weight > 0.0 {
                    Some(cpnt.fx.clone())
                } else {
                    None
                }
            })
            .collect();
        let weights = weights.drain(..).filter(|&w| w > 0.0).collect();

        let mm = Mixture::new(weights, components).unwrap();
        mm.into()
    }
}

macro_rules! impl_quad_bounds {
    (Column<$xtype:ty, $fxtype:ty, $prtype:ty>) => {
        impl QuadBounds for Column<$xtype, $fxtype, $prtype> {
            fn quad_bounds(&self) -> (f64, f64) {
                let components: Vec<&$fxtype> =
                    self.components.iter().map(|cpnt| &cpnt.fx).collect();

                // NOTE: weights not required because weighting does not
                // affect quad bounds in rv 0.8.0
                let mixture = Mixture::uniform(components).unwrap();
                mixture.quad_bounds()
            }
        }
    };
}

impl_quad_bounds!(Column<f64, Gaussian, Ng>);
impl_quad_bounds!(Column<u32, Poisson, Pg>);

impl QmcEntropy for ColModel {
    fn us_needed(&self) -> usize {
        match self {
            ColModel::Continuous(_) => 1,
            ColModel::Categorical(_) => 1,
            ColModel::Labeler(_) => 2,
            ColModel::Count(_) => 1,
        }
    }

    fn q_recip(&self) -> f64 {
        match self {
            ColModel::Categorical(cm) => cm.components()[0].fx.k() as f64,
            ColModel::Continuous(cm) => {
                let (a, b) = cm.quad_bounds();
                b - a
            }
            ColModel::Count(cm) => {
                let (a, b) = cm.quad_bounds();
                b - a
            }
            ColModel::Labeler(cm) => {
                let n_labels = cm.components()[0].fx.n_labels();
                (n_labels * n_labels) as f64
            }
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn us_to_datum(&self, us: &mut Drain<f64>) -> Datum {
        match self {
            ColModel::Continuous(cm) => {
                let (a, b) = cm.quad_bounds();
                let r = b - a;
                let u = us.next().unwrap();
                let x = u.mul_add(r, a);
                debug_assert!(a <= x && x <= b);
                Datum::Continuous(x)
            }
            ColModel::Count(cm) => {
                let (a, b) = cm.quad_bounds();
                let r = b - a;
                let u = us.next().unwrap();
                let x = u.mul_add(r, a).floor() as u32;
                Datum::Count(x)
            }
            ColModel::Categorical(cm) => {
                let k: f64 = cm.components()[0].fx.k() as f64;
                let x = (us.next().unwrap() * k) as u8;
                Datum::Categorical(x)
            }
            ColModel::Labeler(cm) => {
                let n_labels: f64 = cm.components()[0].fx.n_labels() as f64;
                let x1 = (us.next().unwrap() * n_labels) as u8;
                let x2 = (us.next().unwrap() * n_labels) as u8;
                Datum::Label(Label::new(x1, Some(x2)))
            }
        }
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
        col_model.repop_data(data);
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
        col_model.repop_data(data);
        match col_model {
            ColModel::Continuous(ref f) => assert_eq!(f.data.len(), 5),
            _ => panic!("Returned wrong ColModel type."),
        };
    }

    #[test]
    fn to_mixture_with_zero_weight_ignores_component() {
        use approx::*;
        use braid_stats::prior::CsdHyper;
        use rv::data::CategoricalSuffStat;
        use rv::dist::{Categorical, InvGamma, SymmetricDirichlet};

        let col = Column {
            id: 0,
            data: DataContainer {
                data: vec![0_u8, 1_u8, 0_u8],
                present: vec![true, true, true],
            },
            components: vec![
                ConjugateComponent {
                    fx: Categorical::new(&vec![0.1, 0.9]).unwrap(),
                    stat: {
                        let mut stat = CategoricalSuffStat::new(2);
                        stat.observe(&1_u8);
                        stat
                    },
                },
                ConjugateComponent {
                    fx: Categorical::new(&vec![0.8, 0.2]).unwrap(),
                    stat: {
                        let mut stat = CategoricalSuffStat::new(2);
                        stat.observe(&0_u8);
                        stat
                    },
                },
                ConjugateComponent {
                    fx: Categorical::new(&vec![0.3, 0.7]).unwrap(),
                    stat: {
                        let mut stat = CategoricalSuffStat::new(2);
                        stat.observe(&0_u8);
                        stat
                    },
                },
            ],
            prior: Csd {
                symdir: SymmetricDirichlet::new(0.5, 2).unwrap(),
                hyper: CsdHyper {
                    pr_alpha: InvGamma::default(),
                },
            },
        };

        let mm = {
            match col.to_mixture(vec![0.5, 0.0, 0.5]) {
                MixtureType::Categorical(mm) => mm,
                _ => panic!("wrong mixture type"),
            }
        };

        assert_eq!(mm.k(), 2);

        assert_relative_eq!(
            mm.components()[0].weights()[0],
            0.1,
            epsilon = 1E-8
        );
        assert_relative_eq!(
            mm.components()[0].weights()[1],
            0.9,
            epsilon = 1E-8
        );

        assert_relative_eq!(
            mm.components()[1].weights()[0],
            0.3,
            epsilon = 1E-8
        );
        assert_relative_eq!(
            mm.components()[1].weights()[1],
            0.7,
            epsilon = 1E-8
        );
    }
}

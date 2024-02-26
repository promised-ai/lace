use std::mem;
use std::vec::Drain;

use enum_dispatch::enum_dispatch;
use lace_data::{Category, FeatureData};
use lace_data::{Container, SparseContainer};
use lace_stats::assignment::Assignment;
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::rv::data::DataOrSuffStat;
use lace_stats::rv::dist::{
    Bernoulli, Beta, Categorical, Gamma, Gaussian, Mixture,
    NormalInvChiSquared, Poisson, SymmetricDirichlet,
};
use lace_stats::rv::traits::{ConjugatePrior, Mean, QuadBounds, Rv, SuffStat};
use lace_stats::{MixtureType, QmcEntropy};
use lace_utils::MinMax;
use rand::Rng;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::OnceLock;

use super::{Component, MissingNotAtRandom};
use crate::component::ConjugateComponent;
use crate::feature::traits::{Feature, FeatureHelper, TranslateDatum};
use crate::feature::FType;
use crate::traits::{
    AccumScore, LaceDatum, LaceLikelihood, LacePrior, LaceStat,
};
use lace_data::Datum;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
/// A partitioned columns of data
pub struct Column<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    MixtureType: From<Mixture<Fx>>,
    Fx::Stat: LaceStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    pub id: usize,
    pub data: SparseContainer<X>,
    pub components: Vec<ConjugateComponent<X, Fx, Pr>>,
    pub prior: Pr,
    pub hyper: H,
    #[serde(default)]
    pub ignore_hyper: bool,
    #[serde(skip)]
    pub ln_m_cache: OnceLock<<Pr as ConjugatePrior<X, Fx>>::LnMCache>,
}

#[enum_dispatch]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ColModel {
    Continuous(Column<f64, Gaussian, NormalInvChiSquared, NixHyper>),
    Categorical(Column<u8, Categorical, SymmetricDirichlet, CsdHyper>),
    Count(Column<u32, Poisson, Gamma, PgHyper>),
    MissingNotAtRandom(super::mnar::MissingNotAtRandom),
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
            ColModel::MissingNotAtRandom(MissingNotAtRandom { fx, .. }) => {
                fx.impute_bounds()
            }
            _ => None,
        }
    }

    pub fn ftype(&self) -> FType {
        match self {
            Self::Continuous(_) => FType::Continuous,
            Self::Categorical(_) => FType::Categorical,
            Self::Count(_) => FType::Count,
            Self::MissingNotAtRandom(super::mnar::MissingNotAtRandom {
                fx,
                ..
            }) => match &**fx {
                Self::Continuous(_) => FType::Continuous,
                Self::Categorical(_) => FType::Categorical,
                Self::Count(_) => FType::Count,
                _ => panic!("Cannot have mnar of mnar column"),
            },
        }
    }
}

impl<X, Fx, Pr, H> Column<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    MixtureType: From<Mixture<Fx>>,
    Fx::Stat: LaceStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    pub fn new(
        id: usize,
        data: SparseContainer<X>,
        prior: Pr,
        hyper: H,
    ) -> Self {
        Column {
            id,
            data,
            components: Vec::new(),
            ln_m_cache: OnceLock::new(),
            prior,
            hyper,
            ignore_hyper: false,
        }
    }

    #[inline]
    pub fn ln_m_cache(&self) -> &Pr::LnMCache {
        self.ln_m_cache.get_or_init(|| self.prior.ln_m_cache())
    }

    #[inline]
    pub fn unset_ln_m_cache(&mut self) {
        self.ln_m_cache = OnceLock::new();
    }

    pub fn len(&self) -> usize {
        // XXX: this will fail on features with dropped data
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn components(&self) -> &Vec<ConjugateComponent<X, Fx, Pr>> {
        &self.components
    }
}

macro_rules! impl_translate_datum {
    ($x:ty, $fx:ty, $pr:ty, $h:ty, $datum_variant:ident) => {
        impl_translate_datum!($x, $fx, $pr, $h, $datum_variant, $datum_variant);
    };
    ($x:ty, $fx:ty, $pr:ty, $h:ty, $datum_variant:ident, $fdata_variant:ident) => {
        impl TranslateDatum<$x> for Column<$x, $fx, $pr, $h> {
            fn translate_datum(datum: Datum) -> $x {
                match datum {
                    Datum::$datum_variant(x) => x,
                    _ => panic!("Invalid Datum variant for conversion"),
                }
            }

            fn translate_value(x: $x) -> Datum {
                Datum::$datum_variant(x)
            }

            fn translate_feature_data(
                data: FeatureData,
            ) -> SparseContainer<$x> {
                match data {
                    FeatureData::$fdata_variant(xs) => xs,
                    _ => panic!("Invalid FeatureData variant for conversion"),
                }
            }

            fn translate_container(xs: SparseContainer<$x>) -> FeatureData {
                FeatureData::$fdata_variant(xs)
            }

            fn ftype() -> FType {
                FType::$fdata_variant
            }
        }
    };
}

impl TranslateDatum<u8>
    for Column<u8, Categorical, SymmetricDirichlet, CsdHyper>
{
    fn translate_datum(datum: Datum) -> u8 {
        match datum {
            Datum::Categorical(x) => x.as_u8_or_panic(),
            _ => panic!("Invalid Datum variant for conversion"),
        }
    }

    fn translate_value(x: u8) -> Datum {
        Datum::Categorical(Category::U8(x))
    }

    fn translate_feature_data(data: FeatureData) -> SparseContainer<u8> {
        match data {
            FeatureData::Categorical(xs) => xs,
            _ => panic!("Invalid FeatureData variant for conversion"),
        }
    }

    fn translate_container(xs: SparseContainer<u8>) -> FeatureData {
        FeatureData::Categorical(xs)
    }

    fn ftype() -> FType {
        FType::Categorical
    }
}

impl_translate_datum!(bool, Bernoulli, Beta, (), Binary);
impl_translate_datum!(f64, Gaussian, NormalInvChiSquared, NixHyper, Continuous);
impl_translate_datum!(u32, Poisson, Gamma, PgHyper, Count);

pub(crate) fn draw_cpnts<X, Fx, Pr, H>(
    prior: &Pr,
    k: usize,
    mut rng: &mut impl Rng,
) -> Vec<ConjugateComponent<X, Fx, Pr>>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Pr: LacePrior<X, Fx, H>,
    Fx::Stat: LaceStat,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
{
    (0..k)
        .map(|_| ConjugateComponent::new(prior.draw(&mut rng)))
        .collect()
}

#[allow(dead_code)]
impl<X, Fx, Pr, H> Feature for Column<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    Fx::Stat: LaceStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
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
    fn accum_score(&self, scores: &mut [f64], k: usize) {
        // TODO: Decide when to use parallel or serial computation
        self.components[k].accum_score(scores, &self.data);
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
        // re-draw empty k components.
        let mut components = (0..asgn.n_cats)
            .map(|_| {
                ConjugateComponent::new(self.prior.invalid_temp_component())
            })
            .collect::<Vec<_>>();

        // TODO: abstract this away behind the container trait using a
        // zip_with_assignment method.
        self.data.get_slices().iter().for_each(|(ix, xs)| {
            // Creating a sub-slice out of the assignment allows us to bypass
            // bounds checks, which makes things a good deal faster.
            let asgn_sub = unsafe {
                let ptr = asgn.asgn.as_ptr().add(*ix);
                std::slice::from_raw_parts(ptr, xs.len())
            };
            xs.iter()
                .zip(asgn_sub.iter())
                .for_each(|(x, &z)| components[z].observe(x))
        });

        // Set the components
        self.components = components;

        // update the component according to the posterior
        self.update_components(&mut rng);
    }

    #[inline]
    fn score(&self) -> f64 {
        let stats = self.components.iter().map(|cpnt| cpnt.stat.clone());
        self.prior.score_column(stats)
    }

    #[inline]
    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        let empty_stat = self.prior.empty_suffstat();

        let mut stats: Vec<_> =
            (0..asgn.n_cats).map(|_| empty_stat.clone()).collect();

        self.data.get_slices().iter().for_each(|(ix, xs)| {
            // Creating a sub-slice out of the assignment allows us to bypass
            // bounds checks, which makes things a good deal faster.
            let asgn_sub = unsafe {
                let ptr = asgn.asgn.as_ptr().add(*ix);
                std::slice::from_raw_parts(ptr, xs.len())
            };
            xs.iter()
                .zip(asgn_sub.iter())
                .for_each(|(x, &z)| stats[z].observe(x))
        });

        stats.iter().fold(0_f64, |acc, stat| {
            let data = DataOrSuffStat::SuffStat(stat);
            acc + self.prior.ln_m_with_cache(self.ln_m_cache(), &data)
        })
    }

    #[inline]
    fn update_prior_params(&mut self, mut rng: &mut impl Rng) -> f64 {
        if self.ignore_hyper {
            return self
                .components
                .iter()
                .map(|cpnt| self.prior.ln_f(&cpnt.fx))
                .sum::<f64>();
        }

        self.unset_ln_m_cache();
        let components: Vec<&Fx> = self
            .components
            .iter_mut()
            .map(|cpnt| {
                cpnt.reset_ln_pp_cache();
                &cpnt.fx
            })
            .collect();
        self.prior.update_prior(&components, &self.hyper, &mut rng)
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
    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        self.data
            .get(row_ix)
            .map(|x| {
                let cache = self.components[k].ln_pp_cache(&self.prior);
                self.prior.ln_pp_with_cache(cache, &x)
            })
            .unwrap_or(0.0)
    }

    #[inline]
    fn logm(&self, k: usize) -> f64 {
        self.prior
            .ln_m_with_cache(self.ln_m_cache(), &self.components[k].obs())
    }

    #[inline]
    fn singleton_score(&self, row_ix: usize) -> f64 {
        self.data
            .get(row_ix)
            .map(|x| {
                let mut stat = self.prior.empty_suffstat();
                stat.observe(&x);
                self.prior.ln_m_with_cache(
                    self.ln_m_cache(),
                    &DataOrSuffStat::SuffStat(&stat),
                )
            })
            .unwrap_or(0.0)
    }

    #[inline]
    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        if let Some(x) = self.data.get(row_ix) {
            self.components[k].observe(&x);
        }
    }

    #[inline]
    fn take_datum(&mut self, row_ix: usize, k: usize) -> Option<Datum> {
        if let Some(x) = self.data.set_missing(row_ix) {
            self.components[k].forget(&x);
            Some(Self::translate_value(x))
        } else {
            None
        }
    }

    #[inline]
    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        if let Some(x) = self.data.get(row_ix) {
            self.components[k].forget(&x);
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
    fn is_missing(&self, ix: usize) -> bool {
        self.data.is_missing(ix)
    }

    #[inline]
    fn datum(&self, ix: usize) -> Datum {
        self.data
            .get(ix)
            .map(Self::translate_value)
            .unwrap_or(Datum::Missing)
    }

    fn take_data(&mut self) -> FeatureData {
        let mut data: SparseContainer<X> = SparseContainer::default();
        mem::swap(&mut data, &mut self.data);
        Self::translate_container(data)
    }

    fn clone_data(&self) -> FeatureData {
        Self::translate_container(self.data.clone())
    }

    fn draw(&self, k: usize, mut rng: &mut impl Rng) -> Datum {
        let x: X = self.components[k].draw(&mut rng);
        Self::translate_value(x)
    }

    fn repop_data(&mut self, data: FeatureData) {
        let mut xs = Self::translate_feature_data(data);
        mem::swap(&mut xs, &mut self.data);
    }

    fn accum_weights(
        &self,
        datum: &Datum,
        weights: &mut Vec<f64>,
        scaled: bool,
    ) {
        if self.components.len() != weights.len() {
            panic!(
                "Weights: {:?}, n_components: {}",
                weights,
                self.components.len()
            )
        }

        let x: X = Self::translate_datum(datum.clone());

        weights
            .iter_mut()
            .zip(self.components.iter())
            .for_each(|(w, c)| {
                let ln_fx = c.ln_f(&x);
                if scaled {
                    *w += ln_fx - c.fx.ln_f_max().unwrap();
                } else {
                    *w += ln_fx;
                }
            });
    }

    fn accum_exp_weights(&self, datum: &Datum, weights: &mut Vec<f64>) {
        if self.components.len() != weights.len() {
            panic!(
                "Weights: {:?}, n_components: {}",
                weights,
                self.components.len()
            )
        }

        let x: X = Self::translate_datum(datum.clone());

        weights
            .iter_mut()
            .zip(self.components.iter())
            .for_each(|(w, c)| {
                let fx = c.f(&x);
                *w *= fx;
            });
    }

    #[inline]
    fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64 {
        let x: X = Self::translate_datum(datum.clone());
        self.components[k].ln_f(&x)
    }

    #[inline]
    fn cpnt_likelihood(&self, datum: &Datum, k: usize) -> f64 {
        let x: X = Self::translate_datum(datum.clone());
        self.components[k].f(&x)
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
        let weights: Vec<_> = weights.drain(..).filter(|&w| w > 0.0).collect();

        let mm = if weights.is_empty() {
            // If there are no non-zero weights, return an empty mixture
            Mixture::new_unchecked(Vec::new(), Vec::new())
        } else {
            Mixture::new(weights, components).unwrap()
        };

        mm.into()
    }

    fn geweke_init<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) {
        // Draw k components from the prior
        let mut components = (0..asgn.n_cats)
            .map(|_| ConjugateComponent::new(self.prior.draw(rng)))
            .collect::<Vec<_>>();

        // From n data
        let xs: Vec<X> = asgn
            .asgn
            .iter()
            .map(|&zi| {
                let x = components[zi].draw(rng);
                components[zi].observe(&x);
                x
            })
            .collect();

        let data = SparseContainer::from(xs);

        // Set the components
        self.data = data;
        self.components = components;
    }
}

#[allow(dead_code)]
impl<X, Fx, Pr, H> FeatureHelper for Column<X, Fx, Pr, H>
where
    X: LaceDatum,
    Fx: LaceLikelihood<X>,
    Pr: LacePrior<X, Fx, H>,
    H: Serialize + DeserializeOwned,
    Fx::Stat: LaceStat,
    Pr::LnMCache: Clone + std::fmt::Debug,
    Pr::LnPpCache: Send + Sync + Clone + std::fmt::Debug,
    MixtureType: From<Mixture<Fx>>,
    Self: TranslateDatum<X>,
{
    fn del_datum(&mut self, ix: usize) {
        self.data.extract(ix);
    }
}

macro_rules! impl_quad_bounds {
    (Column<$xtype:ty, $fxtype:ty, $prtype:ty, $htype:ty>) => {
        impl QuadBounds for Column<$xtype, $fxtype, $prtype, $htype> {
            fn quad_bounds(&self) -> (f64, f64) {
                let components: Vec<$fxtype> = self
                    .components
                    .iter()
                    .map(|cpnt| cpnt.fx.clone())
                    .collect();

                // NOTE: weights not required because weighting does not
                // affect quad bounds in rv 0.8.0
                let mixture = Mixture::uniform(components).unwrap();
                mixture.quad_bounds()
            }
        }
    };
}

impl_quad_bounds!(Column<f64, Gaussian, NormalInvChiSquared, NixHyper>);
impl_quad_bounds!(Column<u32, Poisson, Gamma, PgHyper>);

impl QmcEntropy for ColModel {
    fn us_needed(&self) -> usize {
        match self {
            ColModel::Continuous(_) => 1,
            ColModel::Categorical(_) => 1,
            ColModel::Count(_) => 1,
            ColModel::MissingNotAtRandom(cm) => cm.fx.us_needed(),
        }
    }

    fn q_recip(&self) -> f64 {
        match self {
            ColModel::MissingNotAtRandom(cm) => cm.fx.q_recip(),
            ColModel::Categorical(cm) => cm.components()[0].fx.k() as f64,
            ColModel::Continuous(cm) => {
                let (a, b) = cm.quad_bounds();
                b - a
            }
            ColModel::Count(cm) => {
                let (a, b) = cm.quad_bounds();
                b - a
            }
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn us_to_datum(&self, us: &mut Drain<f64>) -> Datum {
        match self {
            ColModel::MissingNotAtRandom(cm) => cm.fx.us_to_datum(us),
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
                Datum::Categorical(Category::U8(x))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assignment::AssignmentBuilder;
    use crate::feature::{Column, Feature};
    use lace_data::{FeatureData, SparseContainer};

    use lace_stats::prior::nix::NixHyper;
    fn gauss_fixture() -> ColModel {
        let mut rng = rand::thread_rng();
        let asgn = AssignmentBuilder::new(5)
            .with_alpha(1.0)
            .flat()
            .build()
            .unwrap();
        let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let hyper = NixHyper::default();
        let data = SparseContainer::from(data_vec);
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);

        let mut col = Column::new(0, data, prior, hyper);
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
        let data = SparseContainer::from(data_vec);
        let hyper = CsdHyper::vague(3);
        let prior = hyper.draw(3, &mut rng);

        let mut col = Column::new(0, data, prior, hyper);
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
        use lace_stats::prior::csd::CsdHyper;
        use lace_stats::rv::data::CategoricalSuffStat;
        use lace_stats::rv::dist::{Categorical, SymmetricDirichlet};

        let col = Column {
            id: 0,
            data: SparseContainer::from(vec![
                (0_u8, true),
                (1_u8, true),
                (0_u8, true),
            ]),
            components: vec![
                ConjugateComponent {
                    fx: Categorical::new(&[0.1, 0.9]).unwrap(),
                    stat: {
                        let mut stat = CategoricalSuffStat::new(2);
                        stat.observe(&1_u8);
                        stat
                    },
                    ln_pp_cache: OnceLock::new(),
                },
                ConjugateComponent {
                    fx: Categorical::new(&[0.8, 0.2]).unwrap(),
                    stat: {
                        let mut stat = CategoricalSuffStat::new(2);
                        stat.observe(&0_u8);
                        stat
                    },
                    ln_pp_cache: OnceLock::new(),
                },
                ConjugateComponent {
                    fx: Categorical::new(&[0.3, 0.7]).unwrap(),
                    stat: {
                        let mut stat = CategoricalSuffStat::new(2);
                        stat.observe(&0_u8);
                        stat
                    },
                    ln_pp_cache: OnceLock::new(),
                },
            ],
            prior: SymmetricDirichlet::new_unchecked(0.5, 2),
            hyper: CsdHyper::default(),
            ln_m_cache: OnceLock::new(),
            ignore_hyper: false,
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

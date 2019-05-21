extern crate braid_stats;
extern crate braid_utils;
extern crate rand;
extern crate rv;
extern crate serde;

use std::collections::BTreeMap;
use std::mem;

use braid_stats::prior::{Csd, CsdHyper, Ng, NigHyper};
use braid_utils::misc::minmax;
use rand::Rng;
use rv::dist::{Categorical, Gaussian};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::cc::feature::ColumnGewekeSettings;
use crate::cc::Assignment;
use crate::cc::Column;
use crate::cc::DataContainer;
use crate::cc::Datum;
use crate::cc::FType;
use crate::cc::Feature;
use crate::cc::FeatureData;
use crate::geweke::{GewekeResampleData, GewekeSummarize};
use crate::result;

// TODO: Swap names with Feature.
#[derive(Serialize, Deserialize, Clone)]
pub enum ColModel {
    Continuous(Column<f64, Gaussian, Ng>),
    Categorical(Column<u8, Categorical, Csd>),
    // Binary(Column<bool, Bernoulli, BetaBernoulli),
}

impl ColModel {
    // FIXME: This is a gross mess
    pub fn accum_weights(
        &self,
        datum: &Datum,
        mut weights: Vec<f64>,
    ) -> Vec<f64> {
        match *self {
            ColModel::Continuous(ref ftr) => {
                if ftr.components.len() != weights.len() {
                    let msg = format!(
                        "Weights: {:?}, n_components: {}",
                        weights,
                        ftr.components.len()
                    );
                    panic!(msg);
                }
                let mut accum = |&x| {
                    ftr.components
                        .iter()
                        .enumerate()
                        .for_each(|(k, c)| weights[k] += c.ln_f(x))
                };
                match *datum {
                    Datum::Continuous(ref y) => accum(&y),
                    _ => panic!("Invalid Dtype {:?} for Continuous", datum),
                }
            }
            ColModel::Categorical(ref ftr) => {
                if ftr.components.len() != weights.len() {
                    let msg = format!(
                        "Weights: {:?}, n_components: {}",
                        weights,
                        ftr.components.len()
                    );
                    panic!(msg);
                }
                let mut accum = |x| {
                    ftr.components
                        .iter()
                        .enumerate()
                        .for_each(|(k, c)| weights[k] += c.ln_f(x))
                };
                match *datum {
                    Datum::Categorical(ref y) => accum(y),
                    _ => panic!("Invalid Dtype {:?} for Categorical", datum),
                }
            }
        }
        weights
    }

    pub fn cpnt_logp(&self, datum: &Datum, k: usize) -> f64 {
        match *self {
            ColModel::Continuous(ref ftr) => match *datum {
                Datum::Continuous(ref y) => ftr.components[k].ln_f(y),
                _ => panic!("Invalid Dtype {:?} for Continuous", datum),
            },
            ColModel::Categorical(ref ftr) => match *datum {
                Datum::Categorical(ref y) => ftr.components[k].ln_f(y),
                _ => panic!("Invalid Dtype {:?} for Categorical", datum),
            },
        }
    }

    pub fn draw(&self, k: usize, mut rng: &mut impl Rng) -> Datum {
        match *self {
            ColModel::Continuous(ref ftr) => {
                let x: f64 = ftr.components[k].draw(&mut rng);
                Datum::Continuous(x)
            }
            ColModel::Categorical(ref ftr) => {
                let x: u8 = ftr.components[k].draw(&mut rng);
                Datum::Categorical(x)
            }
        }
    }

    pub fn is_continuous(&self) -> bool {
        match self {
            ColModel::Continuous(_) => true,
            _ => false,
        }
    }

    pub fn is_categorical(&self) -> bool {
        match self {
            ColModel::Categorical(_) => true,
            _ => false,
        }
    }

    pub fn ftype(&self) -> FType {
        match self {
            ColModel::Continuous(_) => FType::Continuous,
            ColModel::Categorical(_) => FType::Categorical,
        }
    }

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

    /// Takes the data out of the column model as `FeatureData` and replaces it
    /// with an empty `DataContainer`.
    pub fn take_data(&mut self) -> FeatureData {
        match self {
            ColModel::Continuous(ftr) => {
                let mut data: DataContainer<f64> = DataContainer::empty();
                mem::swap(&mut data, &mut ftr.data);
                FeatureData::Continuous(data)
            }
            ColModel::Categorical(ftr) => {
                let mut data: DataContainer<u8> = DataContainer::empty();
                mem::swap(&mut data, &mut ftr.data);
                FeatureData::Categorical(data)
            }
        }
    }

    pub fn repop_data(&mut self, data: FeatureData) -> result::Result<()> {
        let err_kind = result::ErrorKind::InvalidDataTypeError;
        match self {
            ColModel::Continuous(ftr) => match data {
                FeatureData::Continuous(mut xs) => {
                    mem::swap(&mut xs, &mut ftr.data);
                    Ok(())
                }
                _ => Err(result::Error::new(
                    err_kind,
                    String::from("Invalid continuous data"),
                )),
            },
            ColModel::Categorical(ftr) => match data {
                FeatureData::Categorical(mut xs) => {
                    mem::swap(&mut xs, &mut ftr.data);
                    Ok(())
                }
                _ => Err(result::Error::new(
                    err_kind,
                    String::from("Invalid categorical data"),
                )),
            },
        }
    }

    pub fn clone_data(&self) -> FeatureData {
        match self {
            ColModel::Continuous(ftr) => {
                let data: DataContainer<f64> = ftr.data.clone();
                FeatureData::Continuous(data)
            }
            ColModel::Categorical(ftr) => {
                let data: DataContainer<u8> = ftr.data.clone();
                FeatureData::Categorical(data)
            }
        }
    }

    pub fn get_datum(&self, row_ix: usize) -> Datum {
        match self {
            ColModel::Continuous(ftr) => {
                if ftr.data.present[row_ix] {
                    Datum::Continuous(ftr.data.data[row_ix])
                } else {
                    Datum::Missing
                }
            }
            ColModel::Categorical(ftr) => {
                if ftr.data.present[row_ix] {
                    Datum::Categorical(ftr.data.data[row_ix])
                } else {
                    Datum::Missing
                }
            }
        }
    }

    pub fn set_id(&mut self, id: usize) {
        match *self {
            ColModel::Continuous(ref mut f) => f.id = id,
            ColModel::Categorical(ref mut f) => f.id = id,
        }
    }
}

impl Feature for ColModel {
    fn id(&self) -> usize {
        match *self {
            ColModel::Continuous(ref f) => f.id,
            ColModel::Categorical(ref f) => f.id,
        }
    }

    fn accum_score(&self, scores: &mut Vec<f64>, k: usize) {
        match *self {
            ColModel::Continuous(ref f) => f.accum_score(scores, k),
            ColModel::Categorical(ref f) => f.accum_score(scores, k),
        }
    }

    fn init_components(&mut self, k: usize, mut rng: &mut impl Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => f.init_components(k, &mut rng),
            ColModel::Categorical(ref mut f) => f.init_components(k, &mut rng),
        }
    }

    fn update_components(&mut self, mut rng: &mut impl Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => f.update_components(&mut rng),
            ColModel::Categorical(ref mut f) => f.update_components(&mut rng),
        }
    }

    fn reassign(&mut self, asgn: &Assignment, mut rng: &mut impl Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => f.reassign(asgn, &mut rng),
            ColModel::Categorical(ref mut f) => f.reassign(asgn, &mut rng),
        }
    }

    fn score(&self) -> f64 {
        match *self {
            ColModel::Continuous(ref f) => f.score(),
            ColModel::Categorical(ref f) => f.score(),
        }
    }

    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        match *self {
            ColModel::Continuous(ref f) => f.asgn_score(asgn),
            ColModel::Categorical(ref f) => f.asgn_score(asgn),
        }
    }

    fn update_prior_params(&mut self, mut rng: &mut impl Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => f.update_prior_params(&mut rng),
            ColModel::Categorical(ref mut f) => f.update_prior_params(&mut rng),
        }
    }

    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => {
                f.append_empty_component(&mut rng)
            }
            ColModel::Categorical(ref mut f) => {
                f.append_empty_component(&mut rng)
            }
        }
    }

    fn drop_component(&mut self, k: usize) {
        match *self {
            ColModel::Continuous(ref mut f) => f.drop_component(k),
            ColModel::Categorical(ref mut f) => f.drop_component(k),
        }
    }

    fn len(&self) -> usize {
        match *self {
            ColModel::Continuous(ref f) => f.len(),
            ColModel::Categorical(ref f) => f.len(),
        }
    }
    fn k(&self) -> usize {
        match *self {
            ColModel::Continuous(ref f) => f.k(),
            ColModel::Categorical(ref f) => f.k(),
        }
    }

    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64> {
        match *self {
            ColModel::Continuous(ref f) => f.logp_at(row_ix, k),
            ColModel::Categorical(ref f) => f.logp_at(row_ix, k),
        }
    }

    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        match *self {
            ColModel::Continuous(ref f) => f.predictive_score_at(row_ix, k),
            ColModel::Categorical(ref f) => f.predictive_score_at(row_ix, k),
        }
    }

    fn singleton_score(&self, row_ix: usize) -> f64 {
        match *self {
            ColModel::Continuous(ref f) => f.singleton_score(row_ix),
            ColModel::Categorical(ref f) => f.singleton_score(row_ix),
        }
    }

    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        match *self {
            ColModel::Continuous(ref mut f) => f.observe_datum(row_ix, k),
            ColModel::Categorical(ref mut f) => f.observe_datum(row_ix, k),
        }
    }

    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        match *self {
            ColModel::Continuous(ref mut f) => f.forget_datum(row_ix, k),
            ColModel::Categorical(ref mut f) => f.forget_datum(row_ix, k),
        }
    }

    fn append_datum(&mut self, x: Datum) {
        match *self {
            ColModel::Continuous(ref mut f) => f.append_datum(x),
            ColModel::Categorical(ref mut f) => f.append_datum(x),
        }
    }
}

// Geweke Trait Implementations
// ============================
impl GewekeSummarize for ColModel {
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> BTreeMap<String, f64> {
        match *self {
            ColModel::Continuous(ref f) => f.geweke_summarize(&settings),
            ColModel::Categorical(ref f) => f.geweke_summarize(&settings),
        }
    }
}

impl GewekeResampleData for ColModel {
    type Settings = ColumnGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&Self::Settings>,
        mut rng: &mut impl Rng,
    ) {
        match *self {
            ColModel::Continuous(ref mut f) => {
                f.geweke_resample_data(settings, &mut rng)
            }
            ColModel::Categorical(ref mut f) => {
                f.geweke_resample_data(settings, &mut rng)
            }
        }
    }
}

pub fn gen_geweke_col_models(
    cm_types: &[FType],
    nrows: usize,
    do_ftr_prior_transition: bool,
    mut rng: &mut impl Rng,
) -> Vec<ColModel> {
    cm_types
        .iter()
        .enumerate()
        .map(|(id, cm_type)| {
            match cm_type {
                FType::Continuous => {
                    let prior = if do_ftr_prior_transition {
                        NigHyper::geweke().draw(&mut rng)
                    } else {
                        Ng::geweke()
                    };
                    // This is filler data, it SHOULD be overwritten at the
                    // start of the geweke run
                    let f = prior.draw(&mut rng);
                    let xs = f.sample(nrows, &mut rng);
                    let data = DataContainer::new(xs);
                    let column = Column::new(id, data, prior);
                    ColModel::Continuous(column)
                }
                FType::Categorical => {
                    let k = 5; // number of categorical values
                    let prior = if do_ftr_prior_transition {
                        CsdHyper::geweke().draw(k, &mut rng)
                    } else {
                        Csd::geweke(k)
                    };
                    let f = prior.draw(&mut rng);
                    let xs = f.sample(nrows, &mut rng);
                    let data = DataContainer::new(xs);
                    let column = Column::new(id, data, prior);
                    ColModel::Categorical(column)
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cc::AssignmentBuilder;
    use crate::cc::Column;

    fn gauss_fixture() -> ColModel {
        let mut rng = rand::thread_rng();
        let asgn = AssignmentBuilder::new(5)
            .with_alpha(1.0)
            .flat()
            .build(&mut rng)
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
            .build(&mut rng)
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

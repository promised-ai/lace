use std::collections::BTreeMap;

use braid_stats::labeler::{Label, Labeler, LabelerPrior};
use braid_stats::prior::{Csd, CsdHyper, Ng, NigHyper};
use braid_utils::misc::minmax;
use enum_dispatch::enum_dispatch;
use rand::Rng;
use rv::dist::{Categorical, Gaussian};
use rv::traits::*;
use serde::{Deserialize, Serialize};

use crate::cc::feature::ColumnGewekeSettings;
use crate::cc::Column;
use crate::cc::DataContainer;
use crate::cc::FType;
use crate::geweke::{GewekeResampleData, GewekeSummarize};

// TODO: Swap names with Feature.
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
            _ => unimplemented!("Unsupported column type"),
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
            _ => unimplemented!("Unsupported column type"),
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
                _ => unimplemented!("Unsupported FType"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cc::container::FeatureData;
    use crate::cc::AssignmentBuilder;
    use crate::cc::Column;
    use crate::cc::Feature;

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

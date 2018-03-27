extern crate rand;

use std::mem;
use std::collections::BTreeMap;
use std::io::{Result, Error, ErrorKind};

use self::rand::Rng;

use misc::{mean, minmax, std};
use cc::Assignment;
use cc::Feature;
use cc::Column;
use cc::DType;
use cc::DataContainer;
use cc::FeatureData;
use dist::prior::{CatSymDirichlet, NormalInverseGamma};
use dist::prior::nig::NigHyper;
use dist::prior::csd::CsdHyper;
use dist::traits::{Distribution, RandomVariate};
use dist::{Categorical, Gaussian};
use geweke::{GewekeResampleData, GewekeSummarize};

// Feature type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FType {
    Continuous,
    Categorical,
}

// TODO: Swap names wiht Feature.
#[derive(Serialize, Deserialize, Clone)]
pub enum ColModel {
    Continuous(Column<f64, Gaussian, NormalInverseGamma>),
    Categorical(Column<u8, Categorical<u8>, CatSymDirichlet>),
    // Binary(Column<bool, Bernoulli, BetaBernoulli),
}

impl ColModel {
    // FIXME: This is a gross mess
    pub fn accum_weights(
        &self,
        datum: &DType,
        mut weights: Vec<f64>,
    ) -> Vec<f64> {
        match *self {
            ColModel::Continuous(ref ftr) => {
                let mut accum = |&x| {
                    ftr.components
                        .iter()
                        .enumerate()
                        .for_each(|(k, c)| weights[k] += c.loglike(x))
                };
                match *datum {
                    DType::Continuous(ref y) => accum(&y),
                    _ => panic!("Invalid Dtype {:?} for Continuous", datum),
                }
            }
            ColModel::Categorical(ref ftr) => {
                let mut accum = |x| {
                    ftr.components
                        .iter()
                        .enumerate()
                        .for_each(|(k, c)| weights[k] += c.loglike(x))
                };
                match *datum {
                    DType::Categorical(ref y) => accum(y),
                    _ => panic!("Invalid Dtype {:?} for Categorical", datum),
                }
            }
        }
        weights
    }

    pub fn cpnt_logp(&self, datum: &DType, k: usize) -> f64 {
        match *self {
            ColModel::Continuous(ref ftr) => match *datum {
                DType::Continuous(ref y) => ftr.components[k].loglike(y),
                _ => panic!("Invalid Dtype {:?} for Continuous", datum),
            },
            ColModel::Categorical(ref ftr) => match *datum {
                DType::Categorical(ref y) => ftr.components[k].loglike(y),
                _ => panic!("Invalid Dtype {:?} for Categorical", datum),
            },
        }
    }

    pub fn draw(&self, k: usize, mut rng: &mut Rng) -> DType {
        match *self {
            ColModel::Continuous(ref ftr) => {
                let x: f64 = ftr.components[k].draw(&mut rng);
                DType::Continuous(x)
            }
            ColModel::Categorical(ref ftr) => {
                let x: u8 = ftr.components[k].draw(&mut rng);
                DType::Categorical(x)
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
                    ftr.components.iter().map(|cpnt| cpnt.mu).collect();
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

    pub fn repop_data(&mut self, data: FeatureData) -> Result<()> {
        let err_kind = ErrorKind::InvalidData;
        match self {
            ColModel::Continuous(ftr) => {
                match data {
                    FeatureData::Continuous(mut xs) => {
                        mem::swap(&mut xs, &mut ftr.data);
                        Ok(())
                    },
                    _ => Err(Error::new(err_kind, "Invalid continuous data"))
                }
            }
            ColModel::Categorical(ftr) => {
                match data {
                    FeatureData::Categorical(mut xs) => {
                        mem::swap(&mut xs, &mut ftr.data);
                        Ok(())
                    },
                    _ => Err(Error::new(err_kind, "Invalid categorical data"))
                }
            }
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

    pub fn get_datum(&self, row_ix: usize) -> DType {
        match self {
            ColModel::Continuous(ftr) => {
                if ftr.data.present[row_ix] {
                    DType::Continuous(ftr.data.data[row_ix])
                } else {
                    DType::Missing
                }
            }
            ColModel::Categorical(ftr) => {
                if ftr.data.present[row_ix] {
                    DType::Categorical(ftr.data.data[row_ix])
                } else {
                    DType::Missing
                }
            }
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

    fn init_components(&mut self, k: usize, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => f.init_components(k, &mut rng),
            ColModel::Categorical(ref mut f) => f.init_components(k, &mut rng),
        }
    }

    fn update_components(&mut self, asgn: &Assignment, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => {
                f.update_components(asgn, &mut rng)
            }
            ColModel::Categorical(ref mut f) => {
                f.update_components(asgn, &mut rng)
            }
        }
    }

    fn reassign(&mut self, asgn: &Assignment, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => {
                f.update_components(asgn, &mut rng)
            }
            ColModel::Categorical(ref mut f) => {
                f.update_components(asgn, &mut rng)
            }
        }
    }

    fn col_score(&self, asgn: &Assignment) -> f64 {
        match *self {
            ColModel::Continuous(ref f) => f.col_score(asgn),
            ColModel::Categorical(ref f) => f.col_score(asgn),
        }
    }

    fn update_prior_params(&mut self, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f) => f.update_prior_params(&mut rng),
            ColModel::Categorical(ref mut f) => f.update_prior_params(&mut rng),
        }
    }

    fn append_empty_component(&mut self, mut rng: &mut Rng) {
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

    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64> {
        match *self {
            ColModel::Continuous(ref f) => f.logp_at(row_ix, k),
            ColModel::Categorical(ref f) => f.logp_at(row_ix, k),
        }
    }
}

// Geweke Trait Implementations
// ============================
impl GewekeSummarize for ColModel {
    fn geweke_summarize(&self) -> BTreeMap<String, f64> {
        match *self {
            ColModel::Continuous(ref f) => geweke_summarize_continuous(f),
            ColModel::Categorical(ref f) => geweke_summarize_categorical(f),
        }
    }
}

impl GewekeResampleData for ColModel {
    type Settings = Assignment;
    fn geweke_resample_data(&mut self, s: Option<&Assignment>, rng: &mut Rng) {
        let asgn = s.unwrap();
        match *self {
            ColModel::Continuous(ref mut f) => {
                for (i, &k) in asgn.asgn.iter().enumerate() {
                    f.data[i] = f.components[k].draw(rng);
                }
            }
            ColModel::Categorical(ref mut f) => {
                for (i, &k) in asgn.asgn.iter().enumerate() {
                    f.data[i] = f.components[k].draw(rng);
                }
            }
        }
    }
}

// Geweke summarizers
// ------------------
fn geweke_summarize_continuous(
    f: &Column<f64, Gaussian, NormalInverseGamma>,
) -> BTreeMap<String, f64> {
    let x_mean = mean(&f.data.data);
    let x_std = std(&f.data.data);

    let mus: Vec<f64> = f.components.iter().map(|c| c.mu).collect();
    let sigmas: Vec<f64> = f.components.iter().map(|c| c.sigma).collect();

    let mu_mean = mean(&mus);
    let sigma_mean = mean(&sigmas);

    let mut stats: BTreeMap<String, f64> = BTreeMap::new();

    stats.insert(String::from("x mean"), x_mean);
    stats.insert(String::from("x std"), x_std);
    stats.insert(String::from("mu mean"), mu_mean);
    stats.insert(String::from("sigma mean"), sigma_mean);

    stats
}

fn geweke_summarize_categorical(
    f: &Column<u8, Categorical<u8>, CatSymDirichlet>,
) -> BTreeMap<String, f64> {
    let x_sum = f.data.data.iter().fold(0, |acc, x| acc + x);

    fn sum_sq(logws: &[f64]) -> f64 {
        logws.iter().fold(0.0, |acc, lw| {
            let w = lw.exp();
            acc + w * w
        })
    }

    let k = f.components.len() as f64;
    let mean_hrm: f64 = f.components
        .iter()
        .fold(0.0, |acc, cpnt| acc + sum_sq(&cpnt.log_weights))
        / k;

    let mut stats: BTreeMap<String, f64> = BTreeMap::new();

    stats.insert(String::from("x sum"), x_sum as f64);
    stats.insert(String::from("weight sum squares"), mean_hrm as f64);

    stats
}

pub fn gen_geweke_col_models(
    cm_types: &[FType],
    nrows: usize,
    mut rng: &mut Rng,
) -> Vec<ColModel> {
    cm_types
        .iter()
        .enumerate()
        .map(|(id, cm_type)| {
            match cm_type {
                FType::Continuous => {
                    let f = Gaussian::new(0.0, 1.0);
                    let xs = f.sample(nrows, &mut rng);
                    let data = DataContainer::new(xs);
                    let prior = NormalInverseGamma::new(
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        NigHyper::geweke(),
                    );
                    let column = Column::new(id, data, prior);
                    ColModel::Continuous(column)
                }
                FType::Categorical => {
                    let k = 5; // number of categorical values
                    let f = Categorical::flat(k);
                    let xs = f.sample(nrows, &mut rng);
                    let data = DataContainer::new(xs);
                    let prior =
                        CatSymDirichlet::new(1.0, k, CsdHyper::geweke());
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

    use dist::prior::{CatSymDirichlet, NormalInverseGamma};
    use cc::Assignment;
    use cc::Column;

    fn gauss_fixture() -> ColModel {
        let mut rng = rand::thread_rng();
        let asgn = Assignment::flat(5, 1.0);
        let data_vec: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let hyper = NigHyper::default();
        let data = DataContainer::new(data_vec);
        let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper);

        let mut col = Column::new(0, data, prior);
        col.reassign(&asgn, &mut rng);
        ColModel::Continuous(col)
    }

    fn categorical_fixture_u8() -> ColModel {
        let mut rng = rand::thread_rng();
        let asgn = Assignment::flat(5, 1.0);
        let data_vec: Vec<u8> = vec![0, 1, 2, 0, 1];
        let data = DataContainer::new(data_vec);
        let prior = CatSymDirichlet::vague(3, &mut rng);

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
}

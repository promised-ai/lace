extern crate rand;

use std::collections::BTreeMap;

use self::rand::Rng;

use misc::{mean, std};
use cc::Assignment;
use cc::Feature;
use cc::Column;
use dist::prior::NormalInverseGamma;
use dist::traits::RandomVariate;
use dist::{Gaussian, Categorical, SymmetricDirichlet};
use geweke::{GewekeResampleData, GewekeSummarize};


// TODO: Swap names wiht Feature.
#[derive(Serialize)]
pub enum ColModel {
    Continuous(Column<f64, Gaussian, NormalInverseGamma>),
    Categorical(Column<u8, Categorical<u8>, SymmetricDirichlet>),
    // Binary(Column<bool, Bernoulli, BetaBernoulli),
}


impl Feature for ColModel {
    fn id(&self) -> usize {
        match *self {
            ColModel::Continuous(ref f)  => f.id,
            ColModel::Categorical(ref f) => f.id,
        }
    }

    fn accum_score(&self, scores: &mut Vec<f64>, k: usize) {
        match *self {
            ColModel::Continuous(ref f)  => f.accum_score(scores, k),
            ColModel::Categorical(ref f) => f.accum_score(scores, k),
        }
    }

    fn update_components(&mut self, asgn: &Assignment, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f)  => f.update_components(asgn, &mut rng),
            ColModel::Categorical(ref mut f) => f.update_components(asgn, &mut rng),
        }
    }

    fn reassign(&mut self, asgn: &Assignment, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f)  => f.update_components(asgn, &mut rng),
            ColModel::Categorical(ref mut f) => f.update_components(asgn, &mut rng),
        }
    }

    fn col_score(&self, asgn: &Assignment) -> f64 {
        match *self {
            ColModel::Continuous(ref f)  => f.col_score(asgn),
            ColModel::Categorical(ref f) => f.col_score(asgn),
        }
    }

    fn update_prior_params(&mut self) {
        unimplemented!();
    }

    fn append_empty_component(&mut self, mut rng: &mut Rng) {
        match *self {
            ColModel::Continuous(ref mut f)  => f.append_empty_component(&mut rng),
            ColModel::Categorical(ref mut f) => f.append_empty_component(&mut rng),
        }
    }

    fn drop_component(&mut self, k: usize) {
        match *self {
            ColModel::Continuous(ref mut f)  => f.drop_component(k),
            ColModel::Categorical(ref mut f) => f.drop_component(k),
        }
    }

    fn len(&self) -> usize {
        match *self {
            ColModel::Continuous(ref f)  => f.len(),
            ColModel::Categorical(ref f) => f.len(),
        }
    }
}


// Geweke Trait Implementations
// ============================
impl GewekeSummarize for ColModel {
    fn geweke_summarize(&self) -> BTreeMap<String, f64> {
        match *self {
            ColModel::Continuous(ref f)  => geweke_summarize_continuous(f),
            ColModel::Categorical(ref f) => geweke_summarize_categorical(f),
        }
    }
}


impl GewekeResampleData for ColModel {
    type Settings = Assignment;
    fn geweke_resample_data(&mut self, s: Option<&Assignment>,
                            rng: &mut Rng) {
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
fn geweke_summarize_continuous(f: &Column<f64, Gaussian, NormalInverseGamma>)
    -> BTreeMap<String, f64>
{
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


fn geweke_summarize_categorical(f: &Column<u8, Categorical<u8>, SymmetricDirichlet>)
    -> BTreeMap<String, f64>
{
    let x_sum = f.data.data.iter().fold(0, |acc, x| acc + x);
    let x_sum_sq = f.data.data.iter().fold(0, |acc, x| acc + x*x);

    let log_weights: Vec<Vec<f64>> = f.components
        .iter()
        .map(|c| c.log_weights.clone())
        .collect();

    let mut stats: BTreeMap<String, f64> = BTreeMap::new();

    stats.insert(String::from("x sum"), x_sum as f64);
    stats.insert(String::from("x sum of squares"), x_sum_sq as f64);

    stats
}

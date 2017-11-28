extern crate serde;
extern crate serde_yaml;
extern crate rand;

use std::collections::BTreeMap;
use std::marker::Sync;

use self::serde::Serialize;
use self::rand::Rng;

use dist::prior::Prior;
use dist::traits::{AccumScore, Distribution};
use cc::container::DataContainer;
use cc::assignment::Assignment;


#[derive(Serialize)]
pub struct Column<T, M, R>
    where T: Clone + Sync,
          M: Distribution<T> + AccumScore<T> + Serialize,
          R: Prior<T, M> + Serialize
{
    pub id: usize,
    #[serde(skip)]
    pub data: DataContainer<T>,
    pub components: Vec<M>,
    pub prior: R,
    // TODO: pointers to data on GPU
}


impl<T, M, R> Column<T, M, R>
    where T: Clone + Sync,
          M: Distribution<T> + AccumScore<T> + Serialize,
          R: Prior<T, M> + Serialize
{
    pub fn new(id: usize, data: DataContainer<T>, prior: R) -> Self {
        Column{id: id, data: data, components: Vec::new(), prior: prior}
    }
}


pub trait Feature {
    fn id(&self) -> usize;
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize);
    fn update_components(&mut self, asgn: &Assignment, rng: &mut Rng);
    fn reassign(&mut self, asgn: &Assignment, rng: &mut Rng);
    fn col_score(&self, asgn: &Assignment) -> f64;
    fn update_prior_params(&mut self);
    fn append_empty_component(&mut self, rng: &mut Rng);
    fn drop_component(&mut self, k: usize);
    fn len(&self) -> usize;

    fn yaml(&self) -> String;

    fn gwk_resample_data(&mut self, asgn: &Assignment, rng: &mut Rng);
    fn gwk_stats(&self) -> BTreeMap<String, f64>;
}


#[allow(dead_code)]
impl<T, M, R> Feature for Column <T, M, R>
    where M: Distribution<T> + AccumScore<T> + Serialize,
          T: Clone + Sync,
          R: Prior<T, M> + Serialize
{
    fn id(&self) -> usize {
        self.id
    }

    fn accum_score(&self, mut scores: &mut Vec<f64>, k: usize) {
        // TODO: Decide when to use parallel or GPU
        self.components[k].accum_score(&mut scores, &self.data.data,
                                       &self.data.present);
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn reassign(&mut self, asgn: & Assignment, mut rng: &mut Rng) {
        self.update_components(&asgn, &mut rng);
    }

    fn update_components(&mut self, asgn: &Assignment, mut rng: &mut Rng) {
        self.components = {
            let mut components: Vec<M> = Vec::with_capacity(asgn.ncats);
            for xk in self.data.group_by(asgn) {
                components.push(self.prior.draw(Some(&xk), &mut rng));
            }
            components
        }
    }

    fn col_score(&self, asgn: &Assignment) -> f64 {
        self.data.group_by(asgn)
                 .iter()
                 .fold(0.0, |acc, xk| acc + self.prior.marginal_score(xk))
    }

    fn update_prior_params(&mut self) {
        self.prior.update_params(&self.components);
    }

    fn append_empty_component(&mut self, mut rng: &mut Rng) {
        self.components.push(self.prior.draw(None, &mut rng));
    }

    fn drop_component(&mut self, k: usize) {
        // cpnt goes out of scope and is dropped 9Hopefully)
        let _cpnt = self.components.remove(k);
    }

    fn yaml(&self) -> String {
        serde_yaml::to_string(&self).unwrap()
    }

    // XXX: This version of reample is only valid for the Finite kernel
    fn gwk_resample_data(&mut self, asgn: &Assignment, rng: &mut Rng) {
        for (i, &k) in asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }

    fn gwk_stats(&self) -> BTreeMap<String, f64> {
        unimplemented!();
    }
}

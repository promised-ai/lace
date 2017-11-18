extern crate rand;

use self::rand::Rng;
use dist::prior::Prior;
use dist::traits::Distribution;
use cc::container::DataContainer;
use cc::assignment::Assignment;


pub struct Column<T, M, R>
    where T: Clone,
          M: Distribution<T>,
          R: Prior<T, M>
{
    pub data: DataContainer<T>,
    pub components: Vec<M>,
    pub prior: R,
    // TODO:
    // - pointers to data on GPU
}


impl<T, M, R> Column<T, M, R> 
    where T: Clone,
          M: Distribution<T>,
          R: Prior<T, M>
{
    pub fn new(data: DataContainer<T>, prior: R) -> Self {
        Column{data: data, components: Vec::new(), prior: prior}
    }
}


pub trait Feature {
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize);
    fn update_components(&mut self, asgn: &Assignment, rng: &mut Rng);
    fn reassign(&mut self, asgn: &Assignment, rng: &mut Rng);
    fn col_score(&self, asgn: &Assignment) -> f64;
    fn update_prior_params(&mut self);
    fn append_empty_component(&mut self, rng: &mut Rng);
    fn drop_component(&mut self, k: usize);
    fn len(&self) -> usize;
}


#[allow(dead_code)]
impl<T, M, R> Feature for Column <T, M, R>
    where M: Distribution<T>,
          T: Clone,
          R: Prior<T, M>
{
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize) {
        // FIXME: account for missing data
        // FIXME: use unnormed log likelihood
        for (i, x) in self.data.data.iter().enumerate() {
            scores[i] += self.components[k].loglike(x);
        }
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
}

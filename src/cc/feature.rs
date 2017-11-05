
use cc::traits::Prior;
use dist::traits::Distribution;
use cc::container::DataContainer;
use cc::assignment::Assignment;


pub struct Feature<'a, T, M, R>
    where T: Clone,
          M: Distribution<T>,
          R: Prior<T, M>
{
    pub data: DataContainer<T>,
    pub asgn: &'a Assignment,
    pub components: Vec<M>,
    pub prior: R,
}

#[allow(dead_code)]
impl<'a, T, M, R> Feature <'a, T, M, R>
    where M: Distribution<T>,
          T: Clone,
          R: Prior<T, M>
{
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize) {
        for (i, x) in self.data.data.iter().enumerate() {
            scores[i] += self.components[k].loglike(x);
        }
    }

    fn update_component_params(&mut self) {
        self.update_components_helper(None);
    }

    fn reassign(&mut self, asgn: &'a Assignment) {
        self.update_components_helper(Some(asgn));
    }

    fn update_components_helper(&mut self, asgn_opt: Option<&'a Assignment>) {
        if let Some(asgn) = asgn_opt {
            self.asgn = asgn;
        }

        self.components = {
            let mut components: Vec<M> = Vec::with_capacity(self.asgn.ncats);
            for xk in self.data.group_by(self.asgn) {
                components.push(self.prior.draw(&xk));
            }
            components
        }

    }

    fn col_score_under_asgn(&self, asgn: &Assignment) -> f64{
        self.data.group_by(asgn)
                 .iter()
                 .fold(0.0, |acc, xk| acc + self.prior.marginal_score(xk))
    }

    fn col_score(&self) -> f64{
        self.col_score_under_asgn(&self.asgn)
    }

    fn update_prior_params(&mut self) {
        self.prior.update_params(&self.components);
    }
}

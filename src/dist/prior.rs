use dist::traits::Distribution;
use dist::Gaussian;
use dist::Bernoulli;


// TODO: rename file to priors
pub trait Prior<T, M>
    where M: Distribution<T>,
{
    fn posterior_draw(&self, data: &Vec<&T>) -> M;
    fn prior_draw(&self) -> M;
    fn marginal_score(&self, y: &Vec<&T>) -> f64;
    fn update_params(&mut self, components: &Vec<M>);

    fn draw(&self, data_opt: Option<&Vec<&T>>) -> M {
        match data_opt {
            Some(data) => self.posterior_draw(data),
            None       => self.prior_draw()
        }
    }

    // Not needed until split-merge or Gibbs implemented:
    // fn predictive_score(&self, x: &T, y: &Vec<T>) -> f64;
    // fn singleton_score(&self, y: &t) -> f64;
}


// Normmal, Inverse-Gamma prior for Normal data
// --------------------------------------------
struct NormalInverseGamma {
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
}


impl NormalInverseGamma {
    fn new(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalInverseGamma{m: m, r: r, s: s, v: v}
    }
}


impl Prior<f64, Gaussian> for NormalInverseGamma {
    fn posterior_draw(&self, data: &Vec<&f64>) -> Gaussian {
        unimplemented!();
    }

    fn prior_draw(&self) -> Gaussian {
        unimplemented!();
    }

    fn marginal_score(&self, y: &Vec<&f64>) -> f64 {
        unimplemented!();
    }

    fn update_params(&mut self, components: &Vec<Gaussian>) {
        unimplemented!();
    }
}


// Beta prior for bernoulli likelihood
// -----------------------------------
struct BetaBernoulli {
    pub a: f64,
    pub b: f64,
}


impl Prior<bool, Bernoulli> for NormalInverseGamma {
    fn posterior_draw(&self, data: &Vec<&bool>) -> Bernoulli {
        unimplemented!();
    }

    fn prior_draw(&self) -> Bernoulli {
        unimplemented!();
    }

    fn marginal_score(&self, y: &Vec<&bool>) -> f64 {
        unimplemented!();
    }

    fn update_params(&mut self, components: &Vec<Bernoulli>) {
        unimplemented!();
    }
}

use dist::traits::Distribution;


pub trait Prior<T, M>
    where M: Distribution<T>,
{
    fn draw(&mut self, data: &Vec<&T>) -> M;
    fn marginal_score(&self, y: &Vec<&T>) -> f64;
    fn update_params(&mut self, &Vec<M>);

    // Not needed until split-merge or Gibbs implemented:
    // fn predictive_score(&self, x: &T, y: &Vec<T>) -> f64;
    // fn singleton_score(&self, y: &t) -> f64;
}

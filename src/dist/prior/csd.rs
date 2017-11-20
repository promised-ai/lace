extern crate num;
extern crate rand;

use std::marker::Sync;
use self::rand::Rng;
use self::num::traits::FromPrimitive;

use special::gammaln;
use misc::bincount;
use dist::Dirichlet;
use dist::traits::RandomVariate;
use dist::SymmetricDirichlet;
use dist::Categorical;
use dist::prior::Prior;


/// Symmetric Dirichlet prior for `Categorical` distribution
impl<T> Prior<T, Categorical<T>> for SymmetricDirichlet
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    fn posterior_draw(&self, data: &Vec<T>,
                      mut rng: &mut Rng) -> Categorical<T>
    {
        let counts = bincount(&data, self.k);
        let alphas = counts.iter().map(|&x| x as f64 + self.alpha).collect();
        let weights = Dirichlet::new(alphas).draw(&mut rng);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Categorical<T> {
        let weights = RandomVariate::draw(self, &mut rng);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn marginal_score(&self, y: &Vec<T>) -> f64 {
        let k = self.k as f64;
        let n = y.len() as f64;
        let counts = bincount(&y, self.k);
        let ak = k * self.alpha;
        let sumg = counts.iter().fold(0.0, |acc, &ct| {
            acc + gammaln(ct as f64 + self.alpha)
        });
        gammaln(ak) - gammaln(ak + n) + sumg - k * gammaln(self.alpha)
    }

    fn update_params(&mut self, components: &Vec<Categorical<T>>) {
        unimplemented!();
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn marginal_likelihood_u8_1() {
        let alpha = 1.0;
        let k = 3;
        let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let symdir = SymmetricDirichlet::new(alpha, k);
        let m = symdir.marginal_score(&xs);

        assert_relative_eq!(-11.3285217419719, m, epsilon=10E-8);
    }

    #[test]
    fn marginal_likelihood_u8_2() {
        let alpha = 0.8;
        let k = 3;
        let mut xs: Vec<u8> = vec![0; 2];
        let mut xs1: Vec<u8> = vec![1; 7];
        let mut xs2: Vec<u8> = vec![2; 13];

        xs.append(&mut xs1);
        xs.append(&mut xs2);

        let symdir = SymmetricDirichlet::new(alpha, k);
        let m = symdir.marginal_score(&xs);

        assert_relative_eq!(-22.4377193008552, m, epsilon=10E-8);
    }

    #[test]
    fn marginal_likelihood_u8_3() {
        let alpha = 4.5;
        let k = 3;
        let mut xs: Vec<u8> = vec![0; 2];
        let mut xs1: Vec<u8> = vec![1; 7];
        let mut xs2: Vec<u8> = vec![2; 13];

        xs.append(&mut xs1);
        xs.append(&mut xs2);

        let symdir = SymmetricDirichlet::new(alpha, k);
        let m = symdir.marginal_score(&xs);

        assert_relative_eq!(-22.4203863897293, m, epsilon=10E-8);
    }
}

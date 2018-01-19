extern crate num;
extern crate rand;

use self::rand::Rng;

use special::gammaln;
use misc::bincount;
use dist::Dirichlet;
use dist::traits::RandomVariate;
use dist::traits::SufficientStatistic;
use dist::traits::Distribution;
use dist::SymmetricDirichlet;
use dist::Categorical;
use dist::categorical::CategoricalSuffStats;
use dist::categorical::CategoricalDatum;
use dist::prior::Prior;


/// Symmetric Dirichlet prior for `Categorical` distribution
impl<T: CategoricalDatum> Prior<T, Categorical<T>> for SymmetricDirichlet {
    fn posterior_draw(&self, data: &[T], mut rng: &mut Rng) -> Categorical<T>
    {
        let mut suffstats = CategoricalSuffStats::new(self.k);
        for x in data {
            suffstats.observe(x);
        }
        // Posterior update weights
        let alphas = suffstats.counts
            .iter()
            .map(|&ct| ct as f64 + self.alpha)
            .collect();
        let weights = Dirichlet::new(alphas).draw(&mut rng);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn loglike(&self, model: &Categorical<T>) -> f64 {
        model.log_weights
            .iter()
            .fold(0.0, |logf, &logw| logf + (self.alpha - 1.0) * logw)
            - self.log_normalizer()
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Categorical<T> {
        let weights = RandomVariate::draw(self, &mut rng);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn marginal_score(&self, y: &[T]) -> f64 {
        let k = self.k as f64;
        let n = y.len() as f64;
        let counts = bincount(y, self.k);
        let ak = k * self.alpha;
        let sumg = counts.iter().fold(0.0, |acc, &ct| {
            acc + gammaln(ct as f64 + self.alpha)
        });
        gammaln(ak) - gammaln(ak + n) + sumg - k * gammaln(self.alpha)
    }

    fn update_params(&mut self, _components: &[Categorical<T>],
                     _rng: &mut Rng)
    {
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

    #[test]
    fn symmetric_prior_draw_log_weights_should_all_be_negative() {
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let ctgrl: Categorical<u8> = symdir.prior_draw(&mut rng);

        assert!(ctgrl.log_weights.iter().all(|lw| *lw < 0.0));
    }

    #[test]
    fn symmetric_prior_draw_log_weights_should_be_unique() {
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let ctgrl: Categorical<u8> = symdir.prior_draw(&mut rng);

        let log_weights = &ctgrl.log_weights;

        assert_relative_ne!(log_weights[0], log_weights[1], epsilon=10e-10);
        assert_relative_ne!(log_weights[1], log_weights[2], epsilon=10e-10);
        assert_relative_ne!(log_weights[2], log_weights[3], epsilon=10e-10);
        assert_relative_ne!(log_weights[0], log_weights[2], epsilon=10e-10);
        assert_relative_ne!(log_weights[0], log_weights[3], epsilon=10e-10);
        assert_relative_ne!(log_weights[1], log_weights[3], epsilon=10e-10);
    }

    #[test]
    fn symmetric_posterior_draw_log_weights_should_all_be_negative() {
        let data: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let ctgrl = symdir.posterior_draw(&data, &mut rng);

        assert!(ctgrl.log_weights.iter().all(|lw| *lw < 0.0));
    }

    #[test]
    fn symmetric_posterior_draw_log_weights_should_be_unique() {
        let data: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
        let mut rng = rand::thread_rng();
        let symdir = SymmetricDirichlet::new(1.0, 4);
        let ctgrl = symdir.posterior_draw(&data, &mut rng);

        let log_weights = &ctgrl.log_weights;

        assert_relative_ne!(log_weights[0], log_weights[1], epsilon=10e-10);
        assert_relative_ne!(log_weights[1], log_weights[2], epsilon=10e-10);
        assert_relative_ne!(log_weights[2], log_weights[3], epsilon=10e-10);
        assert_relative_ne!(log_weights[0], log_weights[2], epsilon=10e-10);
        assert_relative_ne!(log_weights[0], log_weights[3], epsilon=10e-10);
        assert_relative_ne!(log_weights[1], log_weights[3], epsilon=10e-10);
    }
}

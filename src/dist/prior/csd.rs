extern crate num;
extern crate rand;

use self::rand::Rng;

use dist::categorical::CategoricalDatum;
use dist::categorical::CategoricalSuffStats;
use dist::prior::Prior;
use dist::traits::Distribution;
use dist::traits::RandomVariate;
use dist::traits::SufficientStatistic;
use dist::Categorical;
use dist::Dirichlet;
use dist::InvGamma;
use dist::SymmetricDirichlet;
use misc::bincount;
use misc::mh::mh_prior;
use special::gammaln;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatSymDirichlet {
    pub dir: SymmetricDirichlet,
    pub hyper: CsdHyper,
}

impl CatSymDirichlet {
    pub fn new(alpha: f64, k: usize, hyper: CsdHyper) -> Self {
        CatSymDirichlet {
            dir: SymmetricDirichlet::new(alpha, k),
            hyper: hyper,
        }
    }

    pub fn geweke(k: usize) -> Self {
        CatSymDirichlet {
            dir: SymmetricDirichlet::new(1.0, k),
            hyper: CsdHyper::geweke(),
        }
    }

    pub fn from_hyper(
        k: usize,
        hyper: CsdHyper,
        mut rng: &mut impl Rng,
    ) -> Self {
        hyper.draw(k, &mut rng)
    }

    pub fn vague(k: usize, mut rng: &mut impl Rng) -> Self {
        let hyper = CsdHyper::new(k as f64 + 1.0, 1.0);
        hyper.draw(k, &mut rng)
    }
}

/// Symmetric Dirichlet prior for `Categorical` distribution
impl<T: CategoricalDatum> Prior<T, Categorical<T>> for CatSymDirichlet {
    fn posterior_draw(
        &self,
        data: &[T],
        mut rng: &mut impl Rng,
    ) -> Categorical<T> {
        let mut suffstats = CategoricalSuffStats::new(self.dir.k);
        for x in data {
            suffstats.observe(x);
        }
        // Posterior update weights
        let alphas = suffstats
            .counts
            .iter()
            .map(|&ct| ct as f64 + self.dir.alpha)
            .collect();
        let weights = Dirichlet::new(alphas).draw(&mut rng);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn loglike(&self, model: &Categorical<T>) -> f64 {
        model
            .log_weights
            .iter()
            .fold(0.0, |logf, &logw| logf + (self.dir.alpha - 1.0) * logw)
            - self.dir.log_normalizer()
    }

    fn prior_draw(&self, mut rng: &mut impl Rng) -> Categorical<T> {
        let weights = self.dir.draw(&mut rng);
        let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn marginal_score(&self, y: &[T]) -> f64 {
        let k = self.dir.k as f64;
        let n = y.len() as f64;
        let counts = bincount(y, self.dir.k);
        let ak = k * self.dir.alpha;
        let sumg = counts
            .iter()
            .fold(0.0, |acc, &ct| acc + gammaln(ct as f64 + self.dir.alpha));
        gammaln(ak) - gammaln(ak + n) + sumg - k * gammaln(self.dir.alpha)
    }

    fn predictive_score(&self, x: &T, y: &[T]) -> f64 {
        // XXX: The bincount is slow.
        let k = self.dir.k as f64;
        let n = y.len() as f64;
        let counts = bincount(y, self.dir.k);
        let ix: usize = (*x).clone().into();
        let ct_x = counts[ix] as f64;
        (self.dir.alpha + ct_x).ln() - (self.dir.alpha * k + n).ln()
    }

    fn update_params<R: Rng>(
        &mut self,
        components: &[Categorical<T>],
        mut rng: &mut R,
    ) {
        let new_alpha = {
            let draw = |mut rng: &mut R| self.hyper.pr_alpha.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |alpha: &f64| {
                let h = self.hyper.clone();
                let csd = CatSymDirichlet::new(*alpha, self.dir.k, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + csd.loglike(cpnt))
            };
            mh_prior(self.dir.alpha, f, draw, 50, &mut rng)
        };
        self.dir.alpha = new_alpha;
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CsdHyper {
    pub pr_alpha: InvGamma,
}

impl Default for CsdHyper {
    fn default() -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(1.0, 1.0),
        }
    }
}

impl CsdHyper {
    pub fn new(shape: f64, rate: f64) -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(shape, rate),
        }
    }

    pub fn geweke() -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(30.0, 29.0),
        }
    }

    pub fn vague(k: usize) -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(k as f64 + 1.0, 1.0),
        }
    }

    pub fn draw(&self, k: usize, mut rng: &mut impl Rng) -> CatSymDirichlet {
        // SymmetricDirichlet::new(self.pr_alpha.draw(&mut rng), k);
        let alpha = self.pr_alpha.draw(&mut rng);
        CatSymDirichlet::new(alpha, k, self.clone())
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

        let csd = CatSymDirichlet::new(alpha, k, CsdHyper::default());
        let m = csd.marginal_score(&xs);

        assert_relative_eq!(-11.3285217419719, m, epsilon = 10E-8);
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

        let csd = CatSymDirichlet::new(alpha, k, CsdHyper::default());
        let m = csd.marginal_score(&xs);

        assert_relative_eq!(-22.4377193008552, m, epsilon = 10E-8);
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

        let csd = CatSymDirichlet::new(alpha, k, CsdHyper::default());
        let m = csd.marginal_score(&xs);

        assert_relative_eq!(-22.4203863897293, m, epsilon = 10E-8);
    }

    #[test]
    fn symmetric_prior_draw_log_weights_should_all_be_negative() {
        let mut rng = rand::thread_rng();
        let csd = CatSymDirichlet::new(1.0, 4, CsdHyper::default());
        let ctgrl: Categorical<u8> = csd.prior_draw(&mut rng);

        assert!(ctgrl.log_weights.iter().all(|lw| *lw < 0.0));
    }

    #[test]
    fn symmetric_prior_draw_log_weights_should_be_unique() {
        let mut rng = rand::thread_rng();
        let csd = CatSymDirichlet::new(1.0, 4, CsdHyper::default());
        let ctgrl: Categorical<u8> = csd.prior_draw(&mut rng);

        let log_weights = &ctgrl.log_weights;

        assert_relative_ne!(log_weights[0], log_weights[1], epsilon = 10e-10);
        assert_relative_ne!(log_weights[1], log_weights[2], epsilon = 10e-10);
        assert_relative_ne!(log_weights[2], log_weights[3], epsilon = 10e-10);
        assert_relative_ne!(log_weights[0], log_weights[2], epsilon = 10e-10);
        assert_relative_ne!(log_weights[0], log_weights[3], epsilon = 10e-10);
        assert_relative_ne!(log_weights[1], log_weights[3], epsilon = 10e-10);
    }

    #[test]
    fn symmetric_posterior_draw_log_weights_should_all_be_negative() {
        let data: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
        let mut rng = rand::thread_rng();
        let csd = CatSymDirichlet::new(1.0, 4, CsdHyper::default());
        let ctgrl = csd.posterior_draw(&data, &mut rng);

        assert!(ctgrl.log_weights.iter().all(|lw| *lw < 0.0));
    }

    #[test]
    fn symmetric_posterior_draw_log_weights_should_be_unique() {
        let data: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
        let mut rng = rand::thread_rng();
        let csd = CatSymDirichlet::new(1.0, 4, CsdHyper::default());
        let ctgrl = csd.posterior_draw(&data, &mut rng);

        let log_weights = &ctgrl.log_weights;

        assert_relative_ne!(log_weights[0], log_weights[1], epsilon = 10e-10);
        assert_relative_ne!(log_weights[1], log_weights[2], epsilon = 10e-10);
        assert_relative_ne!(log_weights[2], log_weights[3], epsilon = 10e-10);
        assert_relative_ne!(log_weights[0], log_weights[2], epsilon = 10e-10);
        assert_relative_ne!(log_weights[0], log_weights[3], epsilon = 10e-10);
        assert_relative_ne!(log_weights[1], log_weights[3], epsilon = 10e-10);
    }

    #[test]
    fn symmetric_posterior_draw_should_work_with_empty_data() {
        let data: Vec<u8> = vec![];
        let mut rng = rand::thread_rng();
        let csd = CatSymDirichlet::new(1.0, 4, CsdHyper::default());
        csd.posterior_draw(&data, &mut rng);
    }

    #[test]
    fn predictive_probability_value_1() {
        let csd = CatSymDirichlet::new(1.0, 3, CsdHyper::default());
        let x: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let lp = csd.predictive_score(&0, &x);
        assert_relative_eq!(lp, -1.87180217690159, epsilon = 10e-8);
    }

    #[test]
    fn predictive_probability_value_2() {
        let csd = CatSymDirichlet::new(1.0, 3, CsdHyper::default());
        let x: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let lp = csd.predictive_score(&1, &x);
        assert_relative_eq!(lp, -0.95551144502744, epsilon = 10e-8);
    }

    #[test]
    fn predictive_probability_value_3() {
        let csd = CatSymDirichlet::new(2.5, 3, CsdHyper::default());
        let x: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let lp = csd.predictive_score(&0, &x);
        assert_relative_eq!(lp, -1.6094379124341, epsilon = 10e-8);
    }

    #[test]
    fn predictive_probability_value_4() {
        let csd = CatSymDirichlet::new(0.25, 3, CsdHyper::default());
        let x: Vec<u8> = vec![
            0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        ];

        let lp = csd.predictive_score(&0, &x);
        assert_relative_eq!(lp, -2.31363492918062, epsilon = 10e-8);
    }

    #[test]
    fn csd_loglike_value_1() {
        let csd = CatSymDirichlet::new(0.5, 3, CsdHyper::default());
        let cat = Categorical::<u8>::new(vec![-2.30258509, -1.60943791, -0.35667494]);
        let ll = csd.loglike(&cat);
        assert_relative_eq!(ll, 0.29647190827409386, epsilon = 10e-10);
    }
}

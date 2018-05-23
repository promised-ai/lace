extern crate num;
extern crate rand;

use self::rand::Rng;

use dist::Categorical;
use dist::Dirichlet;
use dist::InvGamma;
use dist::SymmetricDirichlet;
use dist::categorical::CategoricalDatum;
use dist::categorical::CategoricalSuffStats;
use dist::prior::Prior;
use dist::traits::Distribution;
use dist::traits::RandomVariate;
use dist::traits::SufficientStatistic;
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

    pub fn from_hyper(k: usize, hyper: CsdHyper, mut rng: &mut Rng) -> Self {
        CatSymDirichlet {
            dir: hyper.draw(k, &mut rng),
            hyper: hyper,
        }
    }

    pub fn vague(k: usize, mut rng: &mut Rng) -> Self {
        let hyper = CsdHyper::new(k as f64 + 1.0, 1.0);
        CatSymDirichlet {
            dir: hyper.draw(k, &mut rng),
            hyper: hyper,
        }
    }
}

/// Symmetric Dirichlet prior for `Categorical` distribution
impl<T: CategoricalDatum> Prior<T, Categorical<T>> for CatSymDirichlet {
    fn posterior_draw(&self, data: &[T], mut rng: &mut Rng) -> Categorical<T> {
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
            .fold(0.0, |logf, &logw| {
                logf + (self.dir.alpha - 1.0) * logw
            }) - self.dir.log_normalizer()
    }

    fn prior_draw(&self, mut rng: &mut Rng) -> Categorical<T> {
        let weights = self.dir.draw(&mut rng);
        let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        Categorical::new(log_weights)
    }

    fn marginal_score(&self, y: &[T]) -> f64 {
        let k = self.dir.k as f64;
        let n = y.len() as f64;
        let counts = bincount(y, self.dir.k);
        let ak = k * self.dir.alpha;
        let sumg = counts.iter().fold(0.0, |acc, &ct| {
            acc + gammaln(ct as f64 + self.dir.alpha)
        });
        gammaln(ak) - gammaln(ak + n) + sumg - k * gammaln(self.dir.alpha)
    }

    fn update_params(
        &mut self,
        components: &[Categorical<T>],
        mut rng: &mut Rng,
    ) {
        let new_alpha: f64;
        {
            let draw = |mut rng: &mut Rng| self.hyper.pr_alpha.draw(&mut rng);
            // TODO: don't clone hyper every time f is called!
            let f = |alpha: &f64| {
                let h = self.hyper.clone();
                let csd = CatSymDirichlet::new(*alpha, self.dir.k, h);
                components
                    .iter()
                    .fold(0.0, |logf, cpnt| logf + csd.loglike(cpnt))
            };
            new_alpha = mh_prior(f, draw, 50, &mut rng);
        }
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
            pr_alpha: InvGamma::new(4.0, 4.0),
        }
    }

    pub fn vague(k: usize) -> Self {
        CsdHyper {
            pr_alpha: InvGamma::new(k as f64 + 1.0, 1.0),
        }
    }

    pub fn draw(&self, k: usize, mut rng: &mut Rng) -> SymmetricDirichlet {
        SymmetricDirichlet::new(self.pr_alpha.draw(&mut rng), k)
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
}

use dist::traits::Distribution;
use dist::Bernoulli;
use dist::Gaussian;


pub trait SufficientStatistic<T> {

    type ModelType: Distribution<T>;

    fn new() -> Self;
    fn observe(&mut self, x: &T);
    fn unobserve(&mut self, x: &T);
}


// Gaussian sufficient statistics
// ------------------------------
pub struct GaussianSuffStats {
    pub n: u64,
    pub sum_x: f64,
    pub sum_x_sq: f64,
}


// TODO: use more numerically stable version
impl SufficientStatistic<f64> for GaussianSuffStats {
    type ModelType = Gaussian;

    fn new() -> Self {
        GaussianSuffStats{n: 0, sum_x: 0.0, sum_x_sq: 0.0}
    }

    fn observe(&mut self, x: &f64) {
        self.n += 1;
        self.sum_x += x;
        self.sum_x_sq += x*x;
    }

    fn unobserve(&mut self, x: &f64) {
        self.n -= 1;
        if self.n == 0 {
            self.sum_x = 0.0;
            self.sum_x_sq  = 0.0;
        } else {
            self.sum_x -= x;
            self.sum_x_sq -= x*x;
       }
   }
}


// Bernoulli sufficient statistics
// -------------------------------
pub struct BernoulliSuffStats {
    pub n: u64,
    pub k: u64,
}


impl SufficientStatistic<bool> for BernoulliSuffStats {
    type ModelType = Bernoulli;

    fn new() -> Self {
        BernoulliSuffStats{n: 0, k: 0}
    }

    fn observe(&mut self, x: &bool) {
        self.n += 1;
        self.k += *x as u64;
    }

    fn unobserve(&mut self, x: &bool) {
        self.n -= 1;
        self.k -= *x as u64;
   }
}

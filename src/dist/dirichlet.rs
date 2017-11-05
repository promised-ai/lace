use dist::distribution::ContinuousDistr;
use special::gamma::gammaln;


pub struct SymmetricDirichlet {
    alpha: f64,
    k: usize,
}

impl SymmetricDirichlet {
    fn new(alpha: f64, k: usize) -> SymmetricDirichlet {
        SymmetricDirichlet{alpha: alpha, k: k)
    }

    fn jeffereys(k: usize) {
        SymmetricDirichlet{alpha: 1.0 / k.into(), k: k}
    }
}

impl ContinuousDistr<Vec<f64>> for SymmetricDirichlet {
    fn log_normalizer(&self) -> f64 {
        let kf: f64 = self.k.into();
        lngamma(kf * self.alpha) - kf * lngamma(self.alpha)
    }

    fn unnormed_logpdf(&self, x: &Vec<f64>) -> f64 {
        x.fold(0.0, |logf, &xi| (self.alpha - 1.0) * xi.ln())
    }
}


impl Moments<Vec<f64>, Vec<f64>> for SymmetricDirichlet {
    fn mean(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha * self.k.into();
        vec![self.alpha/sum_alpha; self.k]
    }

    fn var(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha * self.k.into();
        let numer = self.alpha * (sum_alpha - self.alpha);
        let denom = sum_alpha * sum_alpha * (sum_alpha - 1.0)
        vec![numer/denom; self.k]
    }
}


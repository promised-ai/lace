extern crate rand;

use self::rand::Rng;


pub trait RandomVariate<T> {
    fn draw(&self, rng: &mut Rng) -> T;
    fn sample(&self, n: usize, mut rng: &mut Rng) -> Vec<T> {
        // a terrible slow way to do repeated draws
        (0..n).map(|_| self.draw(&mut rng)).collect()
    }
}


pub trait Distribution<T>: RandomVariate<T> {
    fn unnormed_loglike(&self, x: &T) -> f64;
    fn log_normalizer(&self) -> f64;

    fn like(&self, x: &T) -> f64 {
        self.loglike(x).exp()
    }

    fn loglike(&self, x: &T) -> f64 {
        self.unnormed_loglike(x) - self.log_normalizer()
    }
}


pub trait Cdf<T> {
    fn cdf(&self, x: &T) -> f64;
    fn probability(&self, a: &T, b: &T) -> f64 {
        self.cdf(b) - self.cdf(a)
    }
    fn sf(&self, x: &T) -> f64 {
        1.0 - self.cdf(x)
    }
    fn logcdf(&self, x: &T) -> f64 {
        self.cdf(x).ln()
    }
}


pub trait SufficientStatistic<T> {
    fn new() -> Self;
    fn observe(&mut self, x: &T);
    fn unobserve(&mut self, x: &T);
}


pub trait HasSufficientStatistic<T> {
    fn observe(&mut self, x: &T);
    fn unobserve(&mut self, x: &T);
}


pub trait InverseCdf<T> {
    fn invcdf(&self, p: &T) -> f64;
}


// should some of these be Options?
pub trait Moments<MeanType, VarType> {
    fn mean(&self) -> MeanType;
    fn var(&self) -> VarType;
}


pub trait Mode<ModeType> {
    fn mode(&self) -> ModeType;
}


pub trait Entropy {
    fn entropy(&self) -> f64;
}

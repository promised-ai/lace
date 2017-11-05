extern crate rand;
extern crate num;

use std::marker::PhantomData;
use self::rand::Rng;

use dist::traits::RandomVariate;
use dist::traits::Distribution;
use dist::traits::Entropy;
use dist::traits::Mode;
use misc::argmax;
use misc::logsumexp;
use misc::log_pflip;


pub struct Categorical<T>
    where T: Clone + Into<usize>
{
    pub log_weights: Vec<f64>,  // should be normalized
    phantom: PhantomData<T>,
}


impl<T> Categorical<T>
    where T: Clone + Into<usize>
{
    pub fn new(log_weights: Vec<f64>) -> Categorical<T> {
        let lnorm = logsumexp(&log_weights);
        let normed_weights = log_weights.iter().map(|x| x - lnorm).collect();
        Categorical{log_weights: normed_weights, phantom: PhantomData}
    }

    pub fn flat(k: usize) -> Categorical<T> {
        let weight: f64 = -(k as f64).ln();
        let log_weights: Vec<f64> = vec![weight; k];
        Categorical{log_weights: log_weights, phantom: PhantomData}
    }
}


impl<T> RandomVariate<T> for Categorical<T>
    where T: Clone + Into<usize> + From<usize>
{
    // TODO: Implement alias method for sample
    fn draw(&self, mut rng: &mut Rng) -> T {
        log_pflip(self.log_weights.as_slice(), &mut rng).into()
    }
}


impl<T> Distribution<T> for Categorical<T>
    where T: Clone + Into<usize> + From<usize>
{
    fn unnormed_loglike(&self, x: &T) -> f64 {
        // XXX: I hate this clone.
        let ix: usize = (*x).clone().into();
        self.log_weights[ix]
    }

    fn log_normalizer(&self) -> f64 { 0.0 }
}


impl<T> Mode<usize> for Categorical<T>
    where T: Clone + Into<usize>
{
    fn mode(&self) -> usize {
        argmax(&self.log_weights)
    }
}


impl<T> Entropy for Categorical<T>
    where T: Clone + Into<usize>
{
    fn entropy(&self) -> f64 {
        self.log_weights.iter().fold(0.0, |h, &w| h - w.exp()*w)
    }
}

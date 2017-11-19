extern crate rand;
extern crate num;

use std::marker::Sync;
use std::marker::PhantomData;
use self::rand::Rng;

use dist::traits::RandomVariate;
use dist::traits::Distribution;
use dist::traits::AccumScore;
use dist::traits::Entropy;
use dist::traits::Mode;
use misc::argmax;
use misc::logsumexp;
use misc::log_pflip;


pub struct Categorical<T>
    where T: Clone + Into<usize> + Sync
{
    pub log_weights: Vec<f64>,  // should be normalized
    phantom: PhantomData<T>,
}


impl<T> Categorical<T>
    where T: Clone + Into<usize> + Sync
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
    where T: Clone + Into<usize> + From<usize> + Sync
{
    // TODO: Implement alias method for sample
    fn draw(&self, mut rng: &mut Rng) -> T {
        log_pflip(self.log_weights.as_slice(), &mut rng).into()
    }
}


impl<T> Distribution<T> for Categorical<T>
    where T: Clone + Into<usize> + From<usize> + Sync
{
    fn unnormed_loglike(&self, x: &T) -> f64 {
        // XXX: I hate this clone.
        let ix: usize = (*x).clone().into();
        self.log_weights[ix]
    }

    fn log_normalizer(&self) -> f64 { 0.0 }
}


impl<T> AccumScore<T> for Categorical<T> 
    where T: Clone + Into<usize> + From<usize> + Sync {}


impl<T> Mode<usize> for Categorical<T>
    where T: Clone + Into<usize> + Sync
{
    fn mode(&self) -> usize {
        argmax(&self.log_weights)
    }
}


impl<T> Entropy for Categorical<T>
    where T: Clone + Into<usize> + Sync
{
    fn entropy(&self) -> f64 {
        self.log_weights.iter().fold(0.0, |h, &w| h - w.exp()*w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1E-8; 


    #[test]
    fn categorical_new() {
        let ctgrl: Categorical<u8> = Categorical::new(vec![0.0, 0.1, 0.2]);

        assert_relative_eq!(ctgrl.log_weights[0], -1.20194285, epsilon = TOL);
        assert_relative_eq!(ctgrl.log_weights[1], -1.10194285, epsilon = TOL);
        assert_relative_eq!(ctgrl.log_weights[2], -1.00194285, epsilon = TOL);
    }

    #[test]
    fn categorical_flat() {
        let ctgrl: Categorical<u8> = Categorical::flat(3);

        assert_eq!(ctgrl.log_weights.len(), 3);

        assert_relative_eq!(ctgrl.log_weights[0], ctgrl.log_weights[1],
                            epsilon = TOL);
        assert_relative_eq!(ctgrl.log_weights[1], ctgrl.log_weights[2],
                            epsilon = TOL);
    }
}

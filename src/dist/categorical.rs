extern crate rand;
extern crate num;

use std::marker::Sync;
use std::marker::PhantomData;
use self::rand::Rng;
use self::num::traits::FromPrimitive;

use dist::traits::SufficientStatistic;
use dist::traits::HasSufficientStatistic;
use dist::traits::RandomVariate;
use dist::traits::Distribution;
use dist::traits::AccumScore;
use dist::traits::Entropy;
use dist::traits::Mode;
use misc::argmax;
use misc::logsumexp;
use misc::log_pflip;


pub struct Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    pub log_weights: Vec<f64>,  // should be normalized
    pub suffstats: CategoricalSuffStats<T>
}


impl<T> Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    pub fn new(mut log_weights: Vec<f64>) -> Categorical<T> {
        let k = log_weights.len();
        let lnorm = logsumexp(&log_weights);
        for w in log_weights.iter_mut() {
             *w -= lnorm;
        }
        Categorical{log_weights: log_weights,
                    suffstats:   CategoricalSuffStats::new(k)}
    }

    pub fn flat(k: usize) -> Categorical<T> {
        let weight: f64 = -(k as f64).ln();
        let log_weights: Vec<f64> = vec![weight; k];
        Categorical{log_weights: log_weights,
                    suffstats: CategoricalSuffStats::new(k)}
    }
}

pub struct CategoricalSuffStats<T> 
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    pub n: usize,
    pub counts: Vec<usize>,  // TODO: Vec<f64>?
    _phantom: PhantomData<T>,
}


impl<T> CategoricalSuffStats<T> 
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    pub fn new(k: usize) -> Self {
        CategoricalSuffStats{n: 0, counts: vec![0; k], _phantom: PhantomData}
    }
}


impl<T> SufficientStatistic<T> for CategoricalSuffStats<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    fn observe(&mut self, x: &T) {
        let ix = (*x).clone().into();
        self.n += 1;
        self.counts[ix] += 1;
    }

    fn unobserve(&mut self, x: &T) {
        if self.n == 0 {
            panic!("No data to unobserve");
        }

        let ix = (*x).clone().into();
        self.n -= 1;
        self.counts[ix] -= 1;
   }
}


// TODO: make this a macro
impl<T> HasSufficientStatistic<T> for Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    fn observe(&mut self, x: &T) {
        self.suffstats.observe(x);
    }

    fn unobserve(&mut self, x: &T) {
        self.suffstats.unobserve(x);
    }
}


impl<T> RandomVariate<T> for Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    // TODO: Implement alias method for sample
    fn draw(&self, mut rng: &mut Rng) -> T {
        let ix = log_pflip(self.log_weights.as_slice(), &mut rng);
        FromPrimitive::from_usize(ix).unwrap()
    }
}


impl<T> Distribution<T> for Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    fn unnormed_loglike(&self, x: &T) -> f64 {
        // XXX: I hate this clone.
        let ix: usize = (*x).clone().into();
        self.log_weights[ix]
    }

    fn log_normalizer(&self) -> f64 { 0.0 }
}


impl<T> AccumScore<T> for Categorical<T> 
    where T: Clone + Into<usize> + Sync + FromPrimitive {}


impl<T> Mode<usize> for Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
{
    fn mode(&self) -> usize {
        argmax(&self.log_weights)
    }
}


impl<T> Entropy for Categorical<T>
    where T: Clone + Into<usize> + Sync + FromPrimitive
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
    fn new() {
        let ctgrl: Categorical<u8> = Categorical::new(vec![0.0, 0.1, 0.2]);

        assert_relative_eq!(ctgrl.log_weights[0], -1.20194285, epsilon = TOL);
        assert_relative_eq!(ctgrl.log_weights[1], -1.10194285, epsilon = TOL);
        assert_relative_eq!(ctgrl.log_weights[2], -1.00194285, epsilon = TOL);
    }

    #[test]
    fn flat() {
        let ctgrl: Categorical<u8> = Categorical::flat(3);

        assert_eq!(ctgrl.log_weights.len(), 3);

        assert_relative_eq!(ctgrl.log_weights[0], ctgrl.log_weights[1],
                            epsilon = TOL);
        assert_relative_eq!(ctgrl.log_weights[1], ctgrl.log_weights[2],
                            epsilon = TOL);
    }

    #[test]
    fn suffstat_new() {
        let sf: CategoricalSuffStats<u8> = CategoricalSuffStats::new(3);
        assert!(sf.counts.iter().all(|&x| x == 0));
    }

    #[test]
    fn suffstat_observe() {
        let mut sf: CategoricalSuffStats<u8> = CategoricalSuffStats::new(3);
        sf.observe(&0);
        sf.observe(&1);
        sf.observe(&2);
        sf.observe(&0);
        sf.observe(&2);
        sf.observe(&0);

        assert_eq!(sf.counts[0], 3);
        assert_eq!(sf.counts[1], 1);
        assert_eq!(sf.counts[2], 2);
    }

    #[test]
    fn suffstat_unobserve() {
        let mut sf: CategoricalSuffStats<u8> = CategoricalSuffStats{
            n       : 6,
            counts  : vec![3, 2, 1],
            _phantom: PhantomData
        };
        sf.unobserve(&0);
        sf.unobserve(&1);
        sf.unobserve(&2);

        assert_eq!(sf.counts[0], 2);
        assert_eq!(sf.counts[1], 1);
        assert_eq!(sf.counts[2], 0);
    }

    #[test]
    #[should_panic]
    fn unobserve_empty_panics() {
        let mut sf: CategoricalSuffStats<u8> = CategoricalSuffStats::new(3);
        sf.unobserve(&0);
    }

    #[test]
    #[should_panic]
    fn unobserve_missing_count_panics() {
        let mut sf: CategoricalSuffStats<u8> = CategoricalSuffStats::new(3);
        sf.observe(&1);
        sf.unobserve(&0);
    }
}

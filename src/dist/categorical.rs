extern crate num;
extern crate rand;
extern crate serde;

use self::num::traits::FromPrimitive;
use self::rand::Rng;
use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;
use std::marker::Sync;

use dist::traits::AccumScore;
use dist::traits::Argmax;
use dist::traits::Distribution;
use dist::traits::Entropy;
use dist::traits::HasSufficientStatistic;
use dist::traits::KlDivergence;
use dist::traits::Mode;
use dist::traits::RandomVariate;
use dist::traits::SufficientStatistic;
use misc::argmax;
use misc::log_pflip;
use misc::logsumexp;

/// Specified the types of data that can be used in a `Categorical`
/// distribution.
pub trait CategoricalDatum:
    Sized + Into<usize> + TryFrom<usize> + Sync + Clone + FromPrimitive
{
}

impl<T> CategoricalDatum for T where
    T: Clone + Into<usize> + TryFrom<usize> + Sync + Sized + FromPrimitive
{}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Categorical<T: CategoricalDatum> {
    pub log_weights: Vec<f64>, // should be normalized
    pub suffstats: CategoricalSuffStats<T>,
}

impl<T: CategoricalDatum> Categorical<T> {
    pub fn new(mut log_weights: Vec<f64>) -> Categorical<T> {
        let k = log_weights.len();
        let lnorm = logsumexp(&log_weights);
        for w in &mut log_weights {
            *w -= lnorm;
        }
        Categorical {
            log_weights: log_weights,
            suffstats: CategoricalSuffStats::new(k),
        }
    }

    pub fn flat(k: usize) -> Categorical<T> {
        let weight: f64 = -(k as f64).ln();
        let log_weights: Vec<f64> = vec![weight; k];
        Categorical {
            log_weights: log_weights,
            suffstats: CategoricalSuffStats::new(k),
        }
    }
}

// impl Serialize for Categorical<CategoricalDatum>
//     where T: Clone + Into<usize> + Sync + FromPrimitive
// {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//         where S: Serializer
//     {
//         let mut state = serializer.serialize_struct("Categorical", 1)?;
//         state.serialize_field("log_weights", &self.log_weights)?;
//         state.end()
//     }
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CategoricalSuffStats<T: CategoricalDatum> {
    pub n: usize,
    pub counts: Vec<usize>, // TODO: Vec<f64>?
    #[serde(skip)]
    _phantom: PhantomData<T>,
}

impl<T: CategoricalDatum> CategoricalSuffStats<T> {
    pub fn new(k: usize) -> Self {
        CategoricalSuffStats {
            n: 0,
            counts: vec![0; k],
            _phantom: PhantomData,
        }
    }
}

impl<T: CategoricalDatum> Default for CategoricalSuffStats<T> {
    fn default() -> Self {
        CategoricalSuffStats {
            n: 0,
            counts: vec![],
            _phantom: PhantomData,
        }
    }
}

impl<T: CategoricalDatum> SufficientStatistic<T> for CategoricalSuffStats<T> {
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
impl<T: CategoricalDatum> HasSufficientStatistic<T> for Categorical<T> {
    fn observe(&mut self, x: &T) {
        self.suffstats.observe(x);
    }

    fn unobserve(&mut self, x: &T) {
        self.suffstats.unobserve(x);
    }
}

impl<T: CategoricalDatum> RandomVariate<T> for Categorical<T> {
    // TODO: Implement alias method for sample
    fn draw(&self, mut rng: &mut impl Rng) -> T {
        let ix = log_pflip(self.log_weights.as_slice(), &mut rng);
        FromPrimitive::from_usize(ix).unwrap()
    }
}

impl<T: CategoricalDatum> Distribution<T> for Categorical<T> {
    fn unnormed_loglike(&self, x: &T) -> f64 {
        // XXX: I hate this clone.
        let ix: usize = (*x).clone().into();
        self.log_weights[ix]
    }

    fn log_normalizer(&self) -> f64 {
        0.0
    }
}

impl<T: CategoricalDatum> AccumScore<T> for Categorical<T> {}

impl<T: CategoricalDatum> Mode<usize> for Categorical<T> {
    fn mode(&self) -> usize {
        argmax(&self.log_weights)
    }
}

impl<T: CategoricalDatum> Entropy for Categorical<T> {
    fn entropy(&self) -> f64 {
        self.log_weights.iter().fold(0.0, |h, &w| h - w.exp() * w)
    }
}

impl<T: CategoricalDatum> KlDivergence for Categorical<T> {
    fn kl_divergence(&self, other: &Self) -> f64 {
        self.log_weights
            .iter()
            .zip(other.log_weights.iter())
            .fold(0.0, |acc, (p, q)| acc + p.exp() + p - q)
    }
}

impl<T: CategoricalDatum> Argmax for Categorical<T> {
    type Output = T;
    fn argmax(&self) -> T {
        match self.mode().try_into() {
            Ok(x) => x,
            Err(_) => panic!("Could not convert into T"),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate serde_yaml;
    use super::*;
    use dist::categorical::num::Float;

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

        assert_relative_eq!(
            ctgrl.log_weights[0],
            ctgrl.log_weights[1],
            epsilon = TOL
        );
        assert_relative_eq!(
            ctgrl.log_weights[1],
            ctgrl.log_weights[2],
            epsilon = TOL
        );
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
        let mut sf: CategoricalSuffStats<u8> = CategoricalSuffStats {
            n: 6,
            counts: vec![3, 2, 1],
            _phantom: PhantomData,
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

    // #[test]
    // fn serialize_singleton() {
    //     let ctgrl: Categorical<u8> = Categorical::flat(1);
    //     let yaml = serde_yaml::to_string(&ctgrl).unwrap();
    //     assert_eq!(yaml, "---\nlog_weights:\n  - 0");
    // }

    // #[test]
    // fn serialize_2() {
    //     let ctgrl: Categorical<u8> = Categorical::flat(2);
    //     let yaml = serde_yaml::to_string(&ctgrl).unwrap();
    //     let s = "---\nlog_weights:\n  - -0.6931471805599453\n  - -0.6931471805599453";
    //     assert_eq!(yaml, s);
    // }

    #[test]
    fn argmax_of_flat_should_be_0() {
        let ctgrl: Categorical<u8> = Categorical::flat(4);

        assert_eq!(ctgrl.argmax(), 0);
    }

    #[test]
    fn argmax_value_check_at_first_index() {
        let log_weights = [0.6, 0.3, 0.1].iter().map(|w| w.ln());
        let ctgrl: Categorical<u8> = Categorical::new(log_weights.collect());

        assert_eq!(ctgrl.argmax(), 0);
    }

    #[test]
    fn argmax_value_check_at_middle_index() {
        let log_weights = [0.3, 0.6, 0.1].iter().map(|w| w.ln());
        let ctgrl: Categorical<u8> = Categorical::new(log_weights.collect());

        assert_eq!(ctgrl.argmax(), 1);
    }

    #[test]
    fn argmax_value_check_at_last_index() {
        let log_weights = [0.3, 0.1, 0.6].iter().map(|w| w.ln());
        let ctgrl: Categorical<u8> = Categorical::new(log_weights.collect());

        assert_eq!(ctgrl.argmax(), 2);
    }
}
